"""
ml/pipeline/embedder.py

Dense vector encoding using sentence-transformers (all-MiniLM-L6-v2).

The model produces 384-dimensional L2-normalised vectors suitable for
cosine-similarity search in FAISS / ChromaDB.

Features
--------
- Singleton model loading (loaded once per worker process via module cache)
- Batch encoding with configurable batch size
- Automatic GPU use if torch detects CUDA; falls back to CPU
- Max-sequence-length truncation to 256 tokens (model limit)

Usage:
    embedder = Embedder()
    vectors = embedder.encode(["text one", "text two"])
    # vectors.shape == (2, 384)
"""

import logging
import os
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}   # module-level singleton so workers load the model once

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "64"))
MAX_SEQ_LEN = 256


def _load_model(model_name: str):
    """Load (or return cached) SentenceTransformer model."""
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install it: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        model.max_seq_length = MAX_SEQ_LEN
        _MODEL_CACHE[model_name] = model
        logger.info("Model loaded successfully.")

    return _MODEL_CACHE[model_name]


class Embedder:
    """
    Wrapper around a SentenceTransformer model.

    Args:
        model_name: HuggingFace model ID (default: all-MiniLM-L6-v2).
        batch_size: Number of texts to encode in a single forward pass.
        normalize:  L2-normalise output vectors (recommended for cosine search).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize
        self._model     = _load_model(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: List of strings to embed (empty strings are replaced
                   with a single space to avoid model errors).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Replace empty strings — model errors on zero-length inputs
        safe_texts = [t if t.strip() else " " for t in texts]

        logger.debug("Encoding %d texts (batch_size=%d)", len(safe_texts), self.batch_size)

        vectors = self._model.encode(
            safe_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convenience method — encode a single string.

        Returns:
            1-D np.ndarray of shape (embedding_dim,).
        """
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""
        return self._model.get_sentence_embedding_dimension()

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors.
        Since vectors are already normalised, this is just the dot product.
        """
        return float(np.dot(vec_a, vec_b))

    def batch_similarity(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity of *query_vec* against each row of *corpus_vecs*.

        Returns:
            1-D array of similarity scores, shape (n,).
        """
        return corpus_vecs @ query_vec