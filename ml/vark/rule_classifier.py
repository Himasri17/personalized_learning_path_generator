"""
ml/vark/rule_classifier.py
===========================
Phase 1 VARK classifier — deterministic rule-based scoring matrix.

No training data required.  Each VARK style accumulates points based on
thresholds applied to the ``VARKFeatures`` vector.  The style with the
highest total score is returned as the dominant profile.

VARK signal map (from Technical Specification §3.5)
---------------------------------------------------
Visual  (V)  Fast on conceptual Qs, skips text-heavy Qs, lower theory accuracy
Auditory (A) Slower overall, re-reads (high time), excels at narrative theory
Reading (R)  High theory accuracy, long answers, dense text performer
Kinesthetic (K) Fast + accurate on coding, lower pure-theory score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ml.vark.feature_extractor import VARKFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

VARK_LABELS = ("visual", "auditory", "reading", "kinesthetic")


@dataclass
class VARKResult:
    """Output of any VARK classifier."""

    dominant: str                           # one of VARK_LABELS
    scores: Dict[str, float] = field(default_factory=dict)   # raw scores per style
    confidence: float = 0.0                 # [0.0, 1.0] margin-based confidence
    method: str = "rule"                    # "rule" | "ml"

    def __post_init__(self):
        if self.dominant not in VARK_LABELS:
            raise ValueError(f"dominant must be one of {VARK_LABELS}, got {self.dominant!r}")


# ---------------------------------------------------------------------------
# Thresholds  (tunable without retraining)
# ---------------------------------------------------------------------------

# Timing
SLOW_TIME_MS       = 30_000     # avg > 30s → re-reading / auditory signal
FAST_TIME_MS       = 15_000     # avg < 15s → quick scanner / visual or kinesthetic

# Accuracy
HIGH_ACCURACY      = 0.70       # above this = strong signal
LOW_ACCURACY       = 0.40       # below this = weak signal

# Skip
HIGH_SKIP_RATIO    = 0.20       # skips > 20% → impatient / visual skimmer

# Answer length
HIGH_LONG_ANSWER   = 0.50       # > 50% of theory answers are long → Reading/Writing

# Conceptual ratio
HIGH_CONCEPT_RATIO = 0.60       # spends most time on conceptual Qs


# ---------------------------------------------------------------------------
# Scoring rules — each rule returns a delta for one or more styles
# ---------------------------------------------------------------------------

def _score_timing(f: VARKFeatures) -> Dict[str, float]:
    scores: Dict[str, float] = {k: 0.0 for k in VARK_LABELS}
    avg = f.avg_time_per_q_ms

    if avg < FAST_TIME_MS:
        scores["visual"]      += 1.5
        scores["kinesthetic"] += 1.0
    elif avg > SLOW_TIME_MS:
        scores["auditory"]    += 1.5
        scores["reading"]     += 1.0

    # Fast on conceptual specifically → Visual signal
    if f.fast_conceptual:
        scores["visual"] += 1.0

    return scores


def _score_skip(f: VARKFeatures) -> Dict[str, float]:
    scores: Dict[str, float] = {k: 0.0 for k in VARK_LABELS}

    if f.skip_ratio > HIGH_SKIP_RATIO:
        scores["visual"] += 1.5     # skips dense text Qs
    elif f.skip_ratio < 0.05:
        scores["reading"]   += 0.5
        scores["auditory"]  += 0.5

    return scores


def _score_type_accuracy(f: VARKFeatures) -> Dict[str, float]:
    scores: Dict[str, float] = {k: 0.0 for k in VARK_LABELS}

    # Coding accuracy → Kinesthetic
    if f.n_coding > 0:
        if f.coding_accuracy >= HIGH_ACCURACY:
            scores["kinesthetic"] += 2.0
        elif f.coding_accuracy < LOW_ACCURACY:
            scores["kinesthetic"] -= 0.5

    # Theory accuracy → Reading/Writing or Auditory
    if f.n_theory > 0:
        if f.theory_accuracy >= HIGH_ACCURACY:
            scores["reading"]  += 1.5
            scores["auditory"] += 1.0
        elif f.theory_accuracy < LOW_ACCURACY:
            scores["visual"]      += 0.5
            scores["kinesthetic"] += 0.5

    # MCQ accuracy → Visual (scan-and-select pattern)
    if f.n_mcq > 0:
        if f.mcq_accuracy >= HIGH_ACCURACY and f.theory_accuracy < f.mcq_accuracy:
            scores["visual"] += 1.0

    return scores


def _score_answer_behaviour(f: VARKFeatures) -> Dict[str, float]:
    scores: Dict[str, float] = {k: 0.0 for k in VARK_LABELS}

    if f.long_answer_rate > HIGH_LONG_ANSWER:
        scores["reading"] += 2.0
    elif f.long_answer_rate < 0.10 and f.n_theory > 0:
        scores["kinesthetic"] += 0.5
        scores["visual"]      += 0.5

    if f.conceptual_ratio > HIGH_CONCEPT_RATIO:
        scores["auditory"] += 0.5
        scores["reading"]  += 0.5

    return scores


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class RuleClassifier:
    """
    Deterministic VARK classifier using a weighted scoring matrix.

    Each scoring rule contributes to a per-style score.  The dominant
    style is the argmax; confidence is derived from the score margin.

    Usage::

        from ml.vark.rule_classifier import RuleClassifier
        from ml.vark.feature_extractor import FeatureExtractor

        features = FeatureExtractor().extract(quiz_logs)
        result   = RuleClassifier().classify(features)
        print(result.dominant, result.confidence)
    """

    _RULES = [
        _score_timing,
        _score_skip,
        _score_type_accuracy,
        _score_answer_behaviour,
    ]

    def classify(self, features: VARKFeatures) -> VARKResult:
        """
        Apply the rule matrix and return a ``VARKResult``.

        Parameters
        ----------
        features:
            Feature vector produced by ``FeatureExtractor.extract()``.

        Returns
        -------
        VARKResult
            ``dominant`` is the style with highest accumulated score.
            ``confidence`` is the normalised margin between #1 and #2.
        """
        totals: Dict[str, float] = {k: 0.0 for k in VARK_LABELS}

        for rule in self._RULES:
            delta = rule(features)
            for style, pts in delta.items():
                totals[style] += pts

        dominant, confidence = self._argmax_with_confidence(totals)

        logger.debug(
            "RuleClassifier → dominant=%s  confidence=%.2f  scores=%s",
            dominant, confidence, totals,
        )

        return VARKResult(
            dominant=dominant,
            scores=totals,
            confidence=confidence,
            method="rule",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _argmax_with_confidence(scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Return (dominant_style, confidence).

        Confidence = (score_1 - score_2) / (|score_1| + 1e-9)
        Clipped to [0.0, 1.0].
        """
        sorted_styles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_style, best_score = sorted_styles[0]
        _, second_score        = sorted_styles[1] if len(sorted_styles) > 1 else (None, 0.0)

        margin     = best_score - second_score
        confidence = min(1.0, max(0.0, margin / (abs(best_score) + 1e-9)))

        return best_style, confidence