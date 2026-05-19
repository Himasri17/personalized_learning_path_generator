"""
ml/pipeline/vector_store.py

Unified vector-store wrapper.

Backend selection (via VECTOR_STORE_BACKEND env var):
  "faiss"  (default / dev)  — in-process FAISS index, persisted to disk
  "chroma" (prod)           — ChromaDB with persistent storage or HTTP client

Public interface
----------------
  store = VectorStore()
  store.add(collection_name, vectors, texts, metadatas)
  results = store.query(collection_name, query_vector, top_k)
  store.delete_collection(collection_name)

Both backends expose the same interface so the rest of the pipeline
never imports faiss or chromadb directly.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND = os.environ.get("VECTOR_STORE_BACKEND", "faiss").lower()
_FAISS_INDEX_DIR = Path(os.environ.get("FAISS_INDEX_DIR", "/tmp/faiss_indices"))
_CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/tmp/chroma")
_CHROMA_HOST = os.environ.get("CHROMA_HOST")   # set to use HTTP client
_CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class _FAISSBackend:
    """
    Lightweight FAISS wrapper.  Each collection maps to one IndexFlatIP index
    (inner-product ≡ cosine if vectors are L2-normalised) plus a metadata
    pickle file stored alongside the index.

    Files on disk:
      <FAISS_INDEX_DIR>/<collection_name>.index
      <FAISS_INDEX_DIR>/<collection_name>.meta.pkl
    """

    def __init__(self):
        try:
            import faiss as _faiss
            self._faiss = _faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required for the FAISS backend. "
                "Install: pip install faiss-cpu"
            ) from exc
        _FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    def _index_path(self, name: str) -> Path:
        return _FAISS_INDEX_DIR / f"{name}.index"

    def _meta_path(self, name: str) -> Path:
        return _FAISS_INDEX_DIR / f"{name}.meta.pkl"

    def _load_index(self, name: str):
        path = self._index_path(name)
        if path.exists():
            return self._faiss.read_index(str(path))
        return None

    def _load_meta(self, name: str) -> dict:
        path = self._meta_path(name)
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)
        return {"texts": [], "metadatas": []}

    def _save(self, name: str, index, meta: dict) -> None:
        self._faiss.write_index(index, str(self._index_path(name)))
        with open(self._meta_path(name), "wb") as fh:
            pickle.dump(meta, fh)

    def add(
        self,
        collection_name: str,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        dim = vectors.shape[1]
        index = self._load_index(collection_name)
        meta  = self._load_meta(collection_name)

        if index is None:
            # IndexFlatIP — exact inner-product (cosine on normalised vecs)
            index = self._faiss.IndexFlatIP(dim)

        index.add(vectors)
        meta["texts"].extend(texts)
        meta["metadatas"].extend(metadatas)
        self._save(collection_name, index, meta)
        logger.debug("FAISS add: %d vectors → collection '%s'", len(vectors), collection_name)

    def query(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[dict]:
        index = self._load_index(collection_name)
        if index is None or index.ntotal == 0:
            return []

        meta  = self._load_meta(collection_name)
        q = query_vector.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, index.ntotal)

        scores, indices = index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text":     meta["texts"][idx],
                "metadata": meta["metadatas"][idx],
                "score":    float(score),
            })
        return results

    def delete_collection(self, collection_name: str) -> None:
        for path in [self._index_path(collection_name), self._meta_path(collection_name)]:
            if path.exists():
                path.unlink()
        logger.info("FAISS collection '%s' deleted.", collection_name)


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class _ChromaBackend:
    """
    Thin wrapper around ChromaDB's Python client.

    Uses an HTTP client when CHROMA_HOST is set; otherwise uses the
    embedded persistent client (good for single-node prod).
    """

    def __init__(self):
        try:
            import chromadb
            self._chromadb = chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for the Chroma backend. "
                "Install: pip install chromadb"
            ) from exc

        if _CHROMA_HOST:
            self._client = chromadb.HttpClient(host=_CHROMA_HOST, port=_CHROMA_PORT)
            logger.info("ChromaDB HTTP client → %s:%d", _CHROMA_HOST, _CHROMA_PORT)
        else:
            self._client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)
            logger.info("ChromaDB persistent client → %s", _CHROMA_PERSIST_DIR)

    def _get_or_create(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        collection_name: str,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: List[dict],
    ) -> None:
        col = self._get_or_create(collection_name)
        existing = col.count()
        ids = [f"{collection_name}_{existing + i}" for i in range(len(texts))]

        col.add(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        logger.debug("ChromaDB add: %d vectors → collection '%s'", len(vectors), collection_name)

    def query(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[dict]:
        col = self._get_or_create(collection_name)
        if col.count() == 0:
            return []

        top_k = min(top_k, col.count())
        result = col.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            results.append({
                "text":     doc,
                "metadata": meta,
                "score":    1.0 - dist,   # convert distance → similarity
            })
        return results

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name)
            logger.info("ChromaDB collection '%s' deleted.", collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection '%s': %s", collection_name, exc)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Backend-agnostic vector store.

    Selects FAISS or ChromaDB based on VECTOR_STORE_BACKEND env var.
    All callers use this class exclusively.

    Usage:
        store = VectorStore()
        store.add("doc_abc123", vectors, texts, metadatas)
        results = store.query("doc_abc123", query_vec, top_k=5)
    """

    def __init__(self):
        if _BACKEND == "chroma":
            self._backend = _ChromaBackend()
        else:
            self._backend = _FAISSBackend()
        logger.info("VectorStore initialised with backend='%s'", _BACKEND)

    def add(
        self,
        collection_name: str,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        assert len(vectors) == len(texts) == len(metadatas), (
            "vectors, texts, and metadatas must have equal length."
        )
        self._backend.add(collection_name, vectors, texts, metadatas)

    def query(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[dict]:
        """
        Retrieve the *top_k* most relevant chunks for *query_vector*.

        Returns:
            List of dicts: [{"text": str, "metadata": dict, "score": float}]
            Sorted by descending similarity score.
        """
        results = self._backend.query(collection_name, query_vector, top_k)
        return sorted(results, key=lambda r: r["score"], reverse=True)

    def delete_collection(self, collection_name: str) -> None:
        """Remove all vectors and metadata for a collection."""
        self._backend.delete_collection(collection_name)