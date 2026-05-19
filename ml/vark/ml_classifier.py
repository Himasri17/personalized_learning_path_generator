"""
ml/vark/ml_classifier.py
=========================
Phase 2 VARK classifier — trained XGBoost / Random Forest model.

Replaces the rule engine once sufficient labeled user sessions are
available.  The model is trained offline in notebook
``02_vark_classifier_training.ipynb`` and serialised to disk.

At inference time this module:
    1. Loads the saved model from ``MODEL_PATH`` (lazy, thread-safe).
    2. Converts a ``VARKFeatures`` instance to a numeric vector.
    3. Returns a ``VARKResult`` with class probabilities as scores.

Model artefact layout (relative to project root)
-------------------------------------------------
ml/
  models/
    vark_classifier.joblib      ← primary (XGBoost or RF, any sklearn-API)
    vark_label_encoder.joblib   ← LabelEncoder for target class names
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml.vark.feature_extractor import VARKFeatures
from ml.vark.rule_classifier import VARKResult, VARK_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR   = Path(__file__).resolve().parent.parent   # → ml/
MODEL_PATH  = Path(os.getenv("VARK_MODEL_PATH",  str(_BASE_DIR / "models" / "vark_classifier.joblib")))
ENCODER_PATH = Path(os.getenv("VARK_ENCODER_PATH", str(_BASE_DIR / "models" / "vark_label_encoder.joblib")))

# ---------------------------------------------------------------------------
# Lazy loader (singleton, thread-safe)
# ---------------------------------------------------------------------------

_model_lock    = threading.Lock()
_model         = None
_label_encoder = None


def _load_model():
    """Load model + encoder from disk (once). Uses a lock for thread safety."""
    global _model, _label_encoder

    if _model is not None:
        return _model, _label_encoder

    with _model_lock:
        if _model is not None:          # double-checked locking
            return _model, _label_encoder

        try:
            import joblib  # noqa: PLC0415 — intentional lazy import

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"VARK model not found at {MODEL_PATH}")
            if not ENCODER_PATH.exists():
                raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")

            _model         = joblib.load(MODEL_PATH)
            _label_encoder = joblib.load(ENCODER_PATH)
            logger.info("VARK ML model loaded from %s", MODEL_PATH)

        except Exception as exc:
            logger.error("Failed to load VARK ML model: %s", exc)
            raise

    return _model, _label_encoder


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MLClassifier:
    """
    Scikit-learn–compatible VARK classifier wrapper.

    The underlying estimator can be XGBoostClassifier, RandomForestClassifier,
    or any estimator that exposes ``predict_proba``.

    Usage::

        from ml.vark.ml_classifier import MLClassifier
        from ml.vark.feature_extractor import FeatureExtractor

        features = FeatureExtractor().extract(quiz_logs)
        result   = MLClassifier().classify(features)

    Falls back gracefully to ``VARKResult(dominant="reading", confidence=0.0)``
    if the model artefact is missing (e.g. during initial deployment before
    any training data exists).
    """

    def classify(self, features: VARKFeatures) -> VARKResult:
        """
        Run the trained model and return a ``VARKResult``.

        Parameters
        ----------
        features:
            Feature vector from ``FeatureExtractor.extract()``.

        Returns
        -------
        VARKResult
            ``scores`` contains class probabilities (sum ≈ 1.0).
            ``confidence`` is the max class probability.
        """
        try:
            model, encoder = _load_model()
        except Exception as exc:
            logger.warning("ML model unavailable (%s); returning fallback result.", exc)
            return self._fallback(features)

        vector = np.array(features.to_ml_vector(), dtype=np.float32).reshape(1, -1)

        try:
            proba: np.ndarray = model.predict_proba(vector)[0]   # shape (n_classes,)
        except Exception as exc:
            logger.error("model.predict_proba failed: %s", exc)
            return self._fallback(features)

        # Map probabilities to style labels via encoder
        class_labels: List[str] = list(encoder.classes_)
        scores: Dict[str, float] = {
            label: float(p) for label, p in zip(class_labels, proba)
        }

        dominant_idx = int(np.argmax(proba))
        dominant     = class_labels[dominant_idx]
        confidence   = float(proba[dominant_idx])

        logger.debug(
            "MLClassifier → dominant=%s  confidence=%.3f  proba=%s",
            dominant, confidence, scores,
        )

        return VARKResult(
            dominant=dominant,
            scores=scores,
            confidence=confidence,
            method="ml",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(features: VARKFeatures) -> VARKResult:
        """Return a low-confidence uniform result when the model is unavailable."""
        uniform = 1.0 / len(VARK_LABELS)
        return VARKResult(
            dominant=VARK_LABELS[0],
            scores={k: uniform for k in VARK_LABELS},
            confidence=0.0,
            method="ml",
        )

    # ------------------------------------------------------------------
    # Training helpers (called from notebook)
    # ------------------------------------------------------------------

    @staticmethod
    def train(
        X: "np.ndarray",
        y: "np.ndarray",
        model_type: str = "xgboost",
        save: bool = True,
    ) -> "sklearn.base.BaseEstimator":  # type: ignore[name-defined]  # noqa: F821
        """
        Train and optionally persist the VARK classifier.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).  Each row should be
            produced by ``VARKFeatures.to_ml_vector()``.
        y:
            Integer-encoded labels aligned with ``VARK_LABELS``.
        model_type:
            ``"xgboost"`` (default) or ``"random_forest"``.
        save:
            If True, serialise model + encoder to ``MODEL_PATH`` /
            ``ENCODER_PATH``.

        Returns
        -------
        Fitted estimator.
        """
        import joblib
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        encoder = LabelEncoder()
        y_enc   = encoder.fit_transform(y)

        if model_type == "xgboost":
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    estimator),
        ])
        pipeline.fit(X, y_enc)

        if save:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, MODEL_PATH)
            joblib.dump(encoder,  ENCODER_PATH)
            logger.info("Saved model → %s | encoder → %s", MODEL_PATH, ENCODER_PATH)

        # Reset in-memory singleton so next classify() reloads from disk
        global _model, _label_encoder
        _model         = None
        _label_encoder = None

        return pipeline

    @staticmethod
    def evaluate(
        X: "np.ndarray",
        y: "np.ndarray",
    ) -> Dict[str, float]:
        """
        Evaluate loaded model on a held-out set.

        Returns
        -------
        dict
            ``accuracy``, ``macro_f1``, ``weighted_f1``.
        """
        from sklearn.metrics import accuracy_score, f1_score

        model, encoder = _load_model()
        y_enc          = encoder.transform(y)
        proba          = model.predict_proba(X)
        y_pred         = proba.argmax(axis=1)

        return {
            "accuracy":    float(accuracy_score(y_enc, y_pred)),
            "macro_f1":    float(f1_score(y_enc, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_enc, y_pred, average="weighted")),
        }