import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, DateTime,
    ForeignKey, Text, JSON, Boolean, Enum
)
from sqlalchemy.orm import relationship

from ..extensions import db


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


GRADING_STATUS_ENUM = Enum(
    "pending",    # submitted, not yet graded
    "graded",     # score written
    "skipped",    # user skipped
    "errored",    # grader threw an exception
    name="grading_status_enum",
    create_type=True,
)


class Response(db.Model):
    """
    One user answer to one Question inside a QuizSession.

    Grading fields are written by the Celery scoring task (grader.py).
    Behavioural signals (time_taken_ms, hint_used, …) are written by the
    frontend at submit time and consumed by the VARK classifier.
    """
    __tablename__ = "responses"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(String(36), primary_key=True, default=_uuid)

    # ── Foreign keys ──────────────────────────────────────────────────────────
    session_id  = Column(
        String(36), ForeignKey("quiz_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    question_id = Column(
        String(36), ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id     = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Raw answer ────────────────────────────────────────────────────────────
    user_answer   = Column(Text, nullable=True)    # null when skipped
    skipped       = Column(Boolean, default=False, nullable=False)

    # ── Grading ───────────────────────────────────────────────────────────────
    grading_status  = Column(GRADING_STATUS_ENUM, nullable=False, default="pending")
    is_correct      = Column(Boolean, nullable=True)       # True / False / None (theory partial)
    awarded_score   = Column(Float,   nullable=True)       # 0.0 – question.max_score
    similarity_score = Column(Float,  nullable=True)       # cosine sim for theory Qs
    # for coding questions: per-test-case results
    # [{"input": "...", "expected": "...", "actual": "...", "passed": bool}]
    test_results    = Column(JSON, nullable=True)
    grader_feedback = Column(Text, nullable=True)          # brief LLM-generated feedback

    # ── Behavioural signals (VARK features) ───────────────────────────────────
    time_taken_ms   = Column(Integer, nullable=True)       # ms from Q shown → Submit tapped
    hint_used       = Column(Boolean, default=False)       # user opened a hint (if implemented)
    answer_changed  = Column(Boolean, default=False)       # user revised their answer
    tab_switches    = Column(Integer, default=0)           # how many times focus left the tab
    # raw behavioural JSON for future signal expansion
    behaviour_meta  = Column(JSON, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    submitted_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    graded_at    = Column(DateTime(timezone=True), nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    session  = relationship("QuizSession", back_populates="responses")
    question = relationship("Question",    back_populates="responses")
    user     = relationship("User")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def is_graded(self) -> bool:
        return self.grading_status == "graded"

    def mark_graded(
        self,
        is_correct: bool | None,
        awarded_score: float,
        similarity_score: float | None = None,
        test_results: list | None = None,
        feedback: str | None = None,
    ) -> None:
        self.is_correct       = is_correct
        self.awarded_score    = awarded_score
        self.similarity_score = similarity_score
        self.test_results     = test_results
        self.grader_feedback  = feedback
        self.grading_status   = "graded"
        self.graded_at        = datetime.now(timezone.utc)

    def mark_skipped(self) -> None:
        self.skipped        = True
        self.user_answer    = None
        self.awarded_score  = 0.0
        self.is_correct     = False
        self.grading_status = "skipped"
        self.graded_at      = datetime.now(timezone.utc)

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self, include_feedback: bool = True) -> dict:
        data = {
            "id":               self.id,
            "session_id":       self.session_id,
            "question_id":      self.question_id,
            "user_id":          self.user_id,
            # answer
            "user_answer":      self.user_answer,
            "skipped":          self.skipped,
            # grading
            "grading_status":   self.grading_status,
            "is_correct":       self.is_correct,
            "awarded_score":    self.awarded_score,
            "similarity_score": self.similarity_score,
            "test_results":     self.test_results,
            # behaviour
            "time_taken_ms":    self.time_taken_ms,
            "hint_used":        self.hint_used,
            "answer_changed":   self.answer_changed,
            "tab_switches":     self.tab_switches,
            # timestamps
            "submitted_at":     self.submitted_at.isoformat(),
            "graded_at":        self.graded_at.isoformat() if self.graded_at else None,
        }
        if include_feedback:
            data["grader_feedback"] = self.grader_feedback
        return data

    def __repr__(self) -> str:
        return (
            f"<Response {self.id[:8]} "
            f"q={self.question_id[:8]} correct={self.is_correct}>"
        )