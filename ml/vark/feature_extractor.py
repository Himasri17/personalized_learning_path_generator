"""
ml/vark/feature_extractor.py
=============================
Derives structured behavioral feature vectors from raw per-question
quiz logs collected by the Quiz Engine.

Each quiz session produces a list of ``QuestionLog`` records (see schema
below).  ``FeatureExtractor.extract()`` aggregates those records into a
single ``VARKFeatures`` dataclass that both the rule-based and ML
classifiers consume.

Signal definitions
------------------
time_taken_ms       : total milliseconds spent on all answered questions
avg_time_per_q      : mean ms per question (answered only)
skip_ratio          : fraction of questions skipped (0.0 – 1.0)
type_accuracy       : per-type accuracy dict  {"mcq": f, "theory": f, "coding": f}
conceptual_ratio    : fraction of time spent on conceptual (vs practical) Qs
coding_accuracy     : accuracy on coding questions specifically
theory_accuracy     : accuracy on theory questions specifically
mcq_accuracy        : accuracy on MCQ questions specifically
fast_conceptual     : True if avg time on conceptual Qs < overall avg
long_answer_rate    : fraction of theory answers > 100 characters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class QuestionLog:
    """
    One row of behavioral data captured by the quiz engine per question.

    Attributes
    ----------
    question_id     : str       Unique question identifier
    question_type   : str       "mcq" | "theory" | "coding"
    question_kind   : str       "conceptual" | "practical"
    time_taken_ms   : int       Wall-clock ms from render → submit (0 if skipped)
    is_correct      : bool      Whether the answer was scored correct
    was_skipped     : bool      True if user clicked Skip / moved on without answering
    answer_length   : int       Character count of the submitted answer (0 if skipped/MCQ)
    """

    question_id: str
    question_type: str          # "mcq" | "theory" | "coding"
    question_kind: str          # "conceptual" | "practical"
    time_taken_ms: int
    is_correct: bool
    was_skipped: bool
    answer_length: int = 0


@dataclass
class VARKFeatures:
    """
    Aggregated feature vector consumed by VARK classifiers.

    All float fields are in [0.0, 1.0] unless noted.
    """

    # ── timing ──────────────────────────────────────────────────────────────
    total_time_ms: float = 0.0
    avg_time_per_q_ms: float = 0.0
    avg_time_conceptual_ms: float = 0.0
    avg_time_practical_ms: float = 0.0
    fast_conceptual: bool = False       # conceptual avg < overall avg

    # ── skipping ─────────────────────────────────────────────────────────────
    skip_ratio: float = 0.0

    # ── per-type accuracy ────────────────────────────────────────────────────
    mcq_accuracy: float = 0.0
    theory_accuracy: float = 0.0
    coding_accuracy: float = 0.0
    overall_accuracy: float = 0.0

    # ── answer behaviour ─────────────────────────────────────────────────────
    long_answer_rate: float = 0.0       # theory answers > LONG_ANSWER_THRESHOLD chars
    conceptual_ratio: float = 0.0       # fraction of time on conceptual Qs

    # ── counts (useful for confidence weighting) ─────────────────────────────
    n_questions: int = 0
    n_answered: int = 0
    n_skipped: int = 0
    n_mcq: int = 0
    n_theory: int = 0
    n_coding: int = 0

    def to_dict(self) -> Dict:
        """Return a plain dict (JSON-serialisable)."""
        return asdict(self)

    def to_ml_vector(self) -> List[float]:
        """
        Return a fixed-length numeric list for ML model input.

        Order must match the feature list used during training
        (see notebook 02_vark_classifier_training.ipynb).
        """
        return [
            self.avg_time_per_q_ms,
            self.avg_time_conceptual_ms,
            self.avg_time_practical_ms,
            float(self.fast_conceptual),
            self.skip_ratio,
            self.mcq_accuracy,
            self.theory_accuracy,
            self.coding_accuracy,
            self.overall_accuracy,
            self.long_answer_rate,
            self.conceptual_ratio,
        ]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONG_ANSWER_THRESHOLD = 100     # characters; theory answer considered "long"
MIN_QUESTIONS = 1               # guard against division by zero


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Converts a list of ``QuestionLog`` records into a ``VARKFeatures`` instance.

    Usage::

        logs: List[QuestionLog] = ...   # from DB or quiz engine
        features = FeatureExtractor().extract(logs)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, logs: List[QuestionLog]) -> VARKFeatures:
        """
        Aggregate raw quiz logs into a ``VARKFeatures`` dataclass.

        Parameters
        ----------
        logs:
            Per-question behavioral records for a single quiz session.

        Returns
        -------
        VARKFeatures
            Populated feature vector; all ratio fields default to 0.0
            when denominator is zero (e.g. no coding questions).
        """
        if not logs:
            logger.warning("FeatureExtractor received empty log list.")
            return VARKFeatures()

        n_questions = len(logs)
        answered    = [l for l in logs if not l.was_skipped]
        skipped     = [l for l in logs if l.was_skipped]
        n_answered  = len(answered)
        n_skipped   = len(skipped)

        # ── per-type buckets ──────────────────────────────────────────────
        mcq_logs     = [l for l in answered if l.question_type == "mcq"]
        theory_logs  = [l for l in answered if l.question_type == "theory"]
        coding_logs  = [l for l in answered if l.question_type == "coding"]

        conceptual_logs = [l for l in answered if l.question_kind == "conceptual"]
        practical_logs  = [l for l in answered if l.question_kind == "practical"]

        # ── timing ───────────────────────────────────────────────────────────
        total_time         = sum(l.time_taken_ms for l in answered)
        avg_time           = total_time / n_answered if n_answered else 0.0
        avg_time_concept   = (
            sum(l.time_taken_ms for l in conceptual_logs) / len(conceptual_logs)
            if conceptual_logs else 0.0
        )
        avg_time_practical = (
            sum(l.time_taken_ms for l in practical_logs) / len(practical_logs)
            if practical_logs else 0.0
        )
        fast_conceptual = (
            avg_time_concept < avg_time
            if conceptual_logs and n_answered else False
        )

        # ── skip ratio ───────────────────────────────────────────────────────
        skip_ratio = n_skipped / n_questions

        # ── per-type accuracy ────────────────────────────────────────────────
        def _accuracy(bucket: List[QuestionLog]) -> float:
            return sum(l.is_correct for l in bucket) / len(bucket) if bucket else 0.0

        mcq_acc     = _accuracy(mcq_logs)
        theory_acc  = _accuracy(theory_logs)
        coding_acc  = _accuracy(coding_logs)
        overall_acc = _accuracy(answered)

        # ── answer behaviour ─────────────────────────────────────────────────
        long_answer_rate = (
            sum(1 for l in theory_logs if l.answer_length > LONG_ANSWER_THRESHOLD)
            / len(theory_logs)
            if theory_logs else 0.0
        )

        conceptual_time_total = sum(l.time_taken_ms for l in conceptual_logs)
        conceptual_ratio = (
            conceptual_time_total / total_time if total_time else 0.0
        )

        return VARKFeatures(
            total_time_ms=float(total_time),
            avg_time_per_q_ms=avg_time,
            avg_time_conceptual_ms=avg_time_concept,
            avg_time_practical_ms=avg_time_practical,
            fast_conceptual=fast_conceptual,
            skip_ratio=skip_ratio,
            mcq_accuracy=mcq_acc,
            theory_accuracy=theory_acc,
            coding_accuracy=coding_acc,
            overall_accuracy=overall_acc,
            long_answer_rate=long_answer_rate,
            conceptual_ratio=conceptual_ratio,
            n_questions=n_questions,
            n_answered=n_answered,
            n_skipped=n_skipped,
            n_mcq=len(mcq_logs),
            n_theory=len(theory_logs),
            n_coding=len(coding_logs),
        )