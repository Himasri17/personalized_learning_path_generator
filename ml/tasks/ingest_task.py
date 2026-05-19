"""
ml/tasks/ingest_task.py

Phase A Celery task: parse → embed → store in FAISS / ChromaDB

Pipeline steps
--------------
1. Download the raw file from S3/MinIO via storage_key
2. Parse text + headings with pdf_parser (PyMuPDF)
3. Chunk & extract NLP metadata via nlp_processor (spaCy)
4. Encode chunks into dense vectors via embedder (all-MiniLM-L6-v2)
5. Persist vectors to FAISS (dev) or ChromaDB (prod)
6. Update the Document DB record with status + chunk_count
7. Fire Phase B (question generation) as a follow-on task
"""

import logging
import os
import tempfile
from typing import List

import boto3
from botocore.exceptions import ClientError
from celery import shared_task

from ml.pipeline.pdf_parser import PDFParser
from ml.pipeline.nlp_processor import NLPProcessor
from ml.pipeline.embedder import Embedder
from ml.pipeline.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_from_storage(storage_key: str, dest_path: str) -> None:
    """Download a file from S3/MinIO to a local temp path."""
    bucket = os.environ["STORAGE_BUCKET"]
    endpoint_url = os.environ.get("MINIO_ENDPOINT_URL")   # None in prod → real S3

    s3_kwargs = {
        "aws_access_key_id":     os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name":           os.environ.get("AWS_REGION", "us-east-1"),
    }
    if endpoint_url:
        s3_kwargs["endpoint_url"] = endpoint_url

    client = boto3.client("s3", **s3_kwargs)
    try:
        client.download_file(bucket, storage_key, dest_path)
        logger.debug("Downloaded s3://%s/%s → %s", bucket, storage_key, dest_path)
    except ClientError as exc:
        raise RuntimeError(f"Failed to download {storage_key}: {exc}") from exc


def _update_document_status(doc_id: str, status: str, **extra) -> None:
    """
    Update the Document ORM record inside a fresh application context.
    Works even though this code runs in a Celery worker (no Flask request ctx).
    """
    from app import create_app
    from app.extensions import db
    from app.models.document import Document

    app = create_app()
    with app.app_context():
        doc = Document.query.get(doc_id)
        if not doc:
            logger.error("Document %s not found; cannot update status.", doc_id)
            return
        doc.status = status
        for key, val in extra.items():
            setattr(doc, key, val)
        db.session.commit()
        logger.info("Document %s status → %s", doc_id, status)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="ml.tasks.ingest_document",
    max_retries=3,
    default_retry_delay=30,   # seconds
    acks_late=True,
    reject_on_worker_lost=True,
)
def ingest_document_task(self, doc_id: str, storage_key: str) -> dict:
    """
    Phase A — Ingest pipeline.

    Args:
        doc_id:      UUID of the Document record in Postgres.
        storage_key: S3/MinIO object key, e.g. "documents/<user_id>/<uuid>.pdf".

    Returns:
        dict with doc_id, chunk_count, and status.
    """
    logger.info("[ingest] Starting doc_id=%s key=%s", doc_id, storage_key)
    _update_document_status(doc_id, "processing")

    try:
        # ── Step 1: Download file ───────────────────────────────────────────
        ext = storage_key.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp_path = tmp.name

        _download_from_storage(storage_key, tmp_path)

        # ── Step 2: Parse text + headings ───────────────────────────────────
        parser = PDFParser()
        pages: List[dict] = parser.parse(tmp_path)
        # pages = [{"page": int, "heading": str|None, "text": str}, ...]
        logger.info("[ingest] Parsed %d pages from doc_id=%s", len(pages), doc_id)

        # ── Step 3: NLP chunking + metadata ────────────────────────────────
        nlp = NLPProcessor()
        chunks: List[dict] = nlp.process(pages)
        # chunks = [{"chunk_id": str, "text": str, "topics": [...], "entities": [...], "heading": str}, ...]
        logger.info("[ingest] Produced %d chunks for doc_id=%s", len(chunks), doc_id)

        if not chunks:
            raise ValueError("No text chunks extracted — document may be empty or image-only.")

        # ── Step 4: Embed chunks ────────────────────────────────────────────
        embedder = Embedder()
        texts = [c["text"] for c in chunks]
        vectors = embedder.encode(texts)
        # vectors: np.ndarray of shape (n_chunks, 384)

        # ── Step 5: Store in vector DB ──────────────────────────────────────
        store = VectorStore()
        metadatas = [
            {
                "doc_id":   doc_id,
                "chunk_id": c["chunk_id"],
                "heading":  c.get("heading", ""),
                "topics":   ",".join(c.get("topics", [])),
                "entities": ",".join(c.get("entities", [])),
            }
            for c in chunks
        ]
        store.add(
            collection_name=f"doc_{doc_id}",
            vectors=vectors,
            texts=texts,
            metadatas=metadatas,
        )
        logger.info("[ingest] Stored %d vectors for doc_id=%s", len(chunks), doc_id)

        # ── Step 6: Update DB ───────────────────────────────────────────────
        _update_document_status(doc_id, "ready", chunk_count=len(chunks))

        # ── Step 7: Trigger Phase B — question generation ──────────────────
        from ml.tasks.qgen_task import generate_questions_task
        generate_questions_task.apply_async(
            args=[doc_id],
            countdown=2,   # small delay so the status write is committed first
        )

        return {"doc_id": doc_id, "chunk_count": len(chunks), "status": "ready"}

    except Exception as exc:
        logger.exception("[ingest] Failed for doc_id=%s: %s", doc_id, exc)
        _update_document_status(doc_id, "failed", error_message=str(exc))

        # Retry with exponential back-off; after max_retries re-raise
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))

    finally:
        # Clean up temp file regardless of success/failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass