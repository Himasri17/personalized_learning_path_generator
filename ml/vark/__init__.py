"""
ml/vark/__init__.py
====================
Public surface for the VARK classification sub-package.

Consumers should import from here, not from the individual modules::

    from ml.vark import classify_vark, VARKResult, VARKFeatures, QuestionLog

The ``classify_vark`` function automatically routes to the ML model when
its artefact is present on disk, falling back to the rule engine otherwise.
This lets Phase 1 (rule) and Phase 2 (ML) coexist during rollout.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ml.vark.feature_extractor import FeatureExtractor, VARKFeatures, QuestionLog
from ml.vark.rule_classifier   import RuleClassifier, VARKResult
from ml.vark.ml_classifier     import MLClassifier, MODEL_PATH

logger = logging.getLogger(__name__)

__all__ = [
    "classify_vark",
    "VARKFeatures",
    "VARKResult",
    "QuestionLog",
    "FeatureExtractor",
    "RuleClassifier",
    "MLClassifier",
]


def classify_vark(
    logs: list,
    *,
    force_rule: bool = False,
    force_ml: bool = False,
) -> VARKResult:
    """
    Classify a learner's VARK style from raw quiz behavioral logs.

    Routing logic
    -------------
    - ``force_rule=True``  → always use the rule engine (Phase 1)
    - ``force_ml=True``    → always use ML model (Phase 2); raises if model missing
    - default              → use ML if model artefact exists, else rule engine

    Parameters
    ----------
    logs:
        List of ``QuestionLog`` instances from one quiz session.
    force_rule:
        Skip ML model even if it is available.
    force_ml:
        Use ML model; error if model artefact is missing.

    Returns
    -------
    VARKResult
        ``.dominant`` — one of "visual", "auditory", "reading", "kinesthetic"
        ``.confidence`` — [0.0, 1.0]
        ``.method``    — "rule" or "ml"
    """
    features = FeatureExtractor().extract(logs)

    use_ml = (not force_rule) and (force_ml or MODEL_PATH.exists())

    if use_ml:
        logger.debug("classify_vark: routing to MLClassifier")
        return MLClassifier().classify(features)

    logger.debug("classify_vark: routing to RuleClassifier")
    return RuleClassifier().classify(features)