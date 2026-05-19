import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, DateTime,
    ForeignKey, Enum, Text, JSON
)
from sqlalchemy.orm import relationship

from ..extensions import db


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ── Enum types (reused by Question / Response) ────────────────────────────────

DIFFICULTY_ENUM = Enum(
    "beginner", "intermediate", "advanced",
    name="difficulty_enum",
    create_type=True,
)

QUESTION_TYPE_ENUM = Enum(
    "mcq", "theory", "coding",
    name="question_type_enum",
    create_type=True,
)

SESSION_STATUS_ENUM = Enum(
    "created",       # session row exists, no doc yet
    "uploading",     # document being uploaded
    "processing",    # ML pipeline running (embed + qgen)
    "ready",         # questions generated, quiz not started
    "in_progress",   # user answered ≥1 question
    "submitted",     # user hit Submit, scoring running
    "completed",     # score + VARK written, explanation available
    "failed",        # pipeline error
    name="session_status_enum",
    create_type=True,
)

VARK_STYLE_ENUM = Enum(
    "visual", "auditory", "reading", "kinesthetic",
    name="vark_style_enum",
    create_type=True,
)


class QuizSession(db.Model):
    __tablename__ = "quiz_sessions"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(String(36), primary_key=True, default=_uuid)

    # ── Owner ─────────────────────────────────────────────────────────────────
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Session config (set at creation, never mutated) ───────────────────────
    subject          = Column(String(255), nullable=False)
    difficulty_level = Column(DIFFICULTY_ENUM, nullable=False, default="intermediate")
    question_type    = Column(QUESTION_TYPE_ENUM, nullable=False, default="mcq")
    question_count   = Column(Integer, nullable=False, default=10)

    # ── Document link ─────────────────────────────────────────────────────────
    doc_id          = Column(String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    doc_storage_key = Column(String(512), nullable=True)   # S3/MinIO path

    # ── Pipeline / lifecycle ──────────────────────────────────────────────────
    status          = Column(SESSION_STATUS_ENUM, nullable=False, default="created", index=True)
    celery_task_id  = Column(String(255), nullable=True)   # track async task
    error_message   = Column(Text, nullable=True)          # human-readable failure reason

    # ── Results (written after scoring) ──────────────────────────────────────
    score             = Column(Float, nullable=True)        # 0.0 – 1.0
    correct_count     = Column(Integer, nullable=True)
    incorrect_count   = Column(Integer, nullable=True)
    skipped_count     = Column(Integer, nullable=True)
    total_time_ms     = Column(Integer, nullable=True)      # wall-clock quiz time

    # ── VARK outcome ──────────────────────────────────────────────────────────
    vark_style        = Column(VARK_STYLE_ENUM, nullable=True)   # dominant style

    # ── Topic-level weakness map  { topic: accuracy_float } ──────────────────
    topic_accuracy    = Column(JSON, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at    = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at    = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)
    started_at    = Column(DateTime(timezone=True), nullable=True)   # first answer recorded
    submitted_at  = Column(DateTime(timezone=True), nullable=True)   # Submit pressed
    completed_at  = Column(DateTime(timezone=True), nullable=True)   # scoring done

    # ── Relationships ─────────────────────────────────────────────────────────
    user          = relationship("User",         back_populates="sessions")
    document      = relationship("Document",     foreign_keys=[doc_id], lazy="select")
    questions     = relationship("Question",     back_populates="session",
                                 order_by="Question.order_index",
                                 lazy="dynamic", cascade="all, delete-orphan")
    responses     = relationship("Response",     back_populates="session",
                                 lazy="dynamic", cascade="all, delete-orphan")
    vark_profiles = relationship("VarkProfile",  back_populates="session",
                                 lazy="dynamic", cascade="all, delete-orphan")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def is_scoreable(self) -> bool:
        return self.status == "submitted"

    @property
    def score_percent(self) -> float | None:
        return round(self.score * 100, 1) if self.score is not None else None

    def mark_started(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc)
            self.status = "in_progress"

    def mark_submitted(self):
        self.submitted_at = datetime.now(timezone.utc)
        self.status = "submitted"

    def mark_completed(self, score: float, correct: int, incorrect: int, skipped: int,
                       vark_style: str | None = None, topic_accuracy: dict | None = None):
        self.score           = score
        self.correct_count   = correct
        self.incorrect_count = incorrect
        self.skipped_count   = skipped
        self.vark_style      = vark_style
        self.topic_accuracy  = topic_accuracy or {}
        self.completed_at    = datetime.now(timezone.utc)
        self.status          = "completed"

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self, include_questions: bool = False, include_responses: bool = False) -> dict:
        data = {
            "id":               self.id,
            "user_id":          self.user_id,
            "subject":          self.subject,
            "difficulty_level": self.difficulty_level,
            "question_type":    self.question_type,
            "question_count":   self.question_count,
            "doc_id":           self.doc_id,
            "status":           self.status,
            "error_message":    self.error_message,
            # results
            "score":            self.score,
            "score_percent":    self.score_percent,
            "correct_count":    self.correct_count,
            "incorrect_count":  self.incorrect_count,
            "skipped_count":    self.skipped_count,
            "total_time_ms":    self.total_time_ms,
            "vark_style":       self.vark_style,
            "topic_accuracy":   self.topic_accuracy,
            # timestamps
            "created_at":   self.created_at.isoformat(),
            "started_at":   self.started_at.isoformat()   if self.started_at   else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        if include_questions:
            data["questions"] = [q.to_dict() for q in self.questions]
        if include_responses:
            data["responses"] = [r.to_dict() for r in self.responses]
        return data

    def __repr__(self) -> str:
        return f"<QuizSession {self.id[:8]} subject={self.subject!r} status={self.status}>"