import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, DateTime,
    ForeignKey, Text, JSON, Boolean, ARRAY
)
from sqlalchemy.orm import relationship

from ..extensions import db
from .quiz_session import DIFFICULTY_ENUM, QUESTION_TYPE_ENUM


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class Question(db.Model):
    """
    One generated question inside a QuizSession.

    MCQ  → content (stem), options (list[str]), correct_answer (one of options)
    Theory → content (open prompt), correct_answer (model answer / key points)
    Coding → content (problem statement), correct_answer (reference solution),
              test_cases (list[{input, expected_output}])
    """
    __tablename__ = "questions"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(String(36), primary_key=True, default=_uuid)

    # ── Parent session ────────────────────────────────────────────────────────
    session_id = Column(
        String(36), ForeignKey("quiz_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # ── Content ───────────────────────────────────────────────────────────────
    content           = Column(Text, nullable=False)           # question stem / prompt
    question_type     = Column(QUESTION_TYPE_ENUM, nullable=False)
    difficulty        = Column(DIFFICULTY_ENUM,    nullable=False)
    order_index       = Column(Integer, nullable=False, default=0)

    # ── MCQ fields ────────────────────────────────────────────────────────────
    # options: ["Option A", "Option B", "Option C", "Option D"]
    options           = Column(JSON,  nullable=True)

    # ── Answer & explanation ──────────────────────────────────────────────────
    correct_answer    = Column(Text, nullable=False)           # ground truth answer
    explanation       = Column(Text, nullable=True)            # LLM-generated rationale

    # ── Coding-specific ───────────────────────────────────────────────────────
    # test_cases: [{"input": "...", "expected_output": "..."}]
    test_cases        = Column(JSON, nullable=True)
    starter_code      = Column(Text, nullable=True)            # scaffold shown to user
    language          = Column(String(32), nullable=True)      # "python", "javascript" …

    # ── RAG provenance ────────────────────────────────────────────────────────
    # which FAISS/Chroma chunk IDs were used to generate this question
    chunk_ids         = Column(JSON, nullable=True)            # list[str]
    source_page_range = Column(JSON, nullable=True)            # [start_page, end_page]

    # ── Taxonomy / tagging ────────────────────────────────────────────────────
    topic_tags        = Column(JSON, nullable=True)            # list[str]  e.g. ["recursion", "trees"]
    bloom_level       = Column(String(32), nullable=True)      # remember / understand / apply …
    is_active         = Column(Boolean, default=True, nullable=False)  # soft-delete

    # ── Grading metadata ──────────────────────────────────────────────────────
    # max_score per question (useful when questions have different weights)
    max_score         = Column(Float, nullable=False, default=1.0)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at        = Column(DateTime(timezone=True), default=_now, nullable=False)

    # ── Relationships ─────────────────────────────────────────────────────────
    session   = relationship("QuizSession", back_populates="questions")
    responses = relationship("Response",    back_populates="question",
                             lazy="dynamic", cascade="all, delete-orphan")

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def is_mcq(self) -> bool:
        return self.question_type == "mcq"

    @property
    def is_theory(self) -> bool:
        return self.question_type == "theory"

    @property
    def is_coding(self) -> bool:
        return self.question_type == "coding"

    def to_dict(
        self,
        include_answer: bool = False,
        include_test_cases: bool = False,
    ) -> dict:
        """
        Default serialisation hides correct_answer and test_cases so they are
        not sent to the frontend during an active quiz.
        Pass include_answer=True on the results endpoint.
        """
        data = {
            "id":               self.id,
            "session_id":       self.session_id,
            "content":          self.content,
            "question_type":    self.question_type,
            "difficulty":       self.difficulty,
            "order_index":      self.order_index,
            "options":          self.options,
            "topic_tags":       self.topic_tags,
            "bloom_level":      self.bloom_level,
            "max_score":        self.max_score,
            # coding helpers always shown (no spoiler risk)
            "starter_code":     self.starter_code,
            "language":         self.language,
        }
        if include_answer:
            data["correct_answer"]    = self.correct_answer
            data["explanation"]       = self.explanation
            data["source_page_range"] = self.source_page_range
        if include_test_cases:
            data["test_cases"] = self.test_cases
        return data

    def __repr__(self) -> str:
        return (
            f"<Question {self.id[:8]} "
            f"type={self.question_type} idx={self.order_index}>"
        )