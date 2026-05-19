"""
ml/tasks/qgen_task.py

Phase B Celery task: RAG retrieval → LLM prompt → store Questions

Pipeline steps
--------------
1. Load document metadata + subject from Postgres
2. Retrieve top-k representative chunks from the vector store (RAG)
3. Call question_generator to build an LLM prompt and parse the JSON response
4. Persist generated Question rows linked to the Document
5. Update Document.question_count
"""

import logging
import os

from celery import shared_task

from ml.pipeline.vector_store import VectorStore
from ml.pipeline.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)

# How many chunks to pull per difficulty tier
_CHUNKS_PER_DIFFICULTY = {
    "easy":   5,
    "medium": 7,
    "hard":   8,
}

# Default questions to generate when no session config is present
_DEFAULT_Q_COUNT = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_document(doc_id: str):
    """Fetch Document ORM object inside a fresh app context."""
    from app import create_app
    from app.models.document import Document

    app = create_app()
    with app.app_context():
        return Document.query.get(doc_id)


def _save_questions(doc_id: str, questions: list) -> int:
    """
    Persist a list of question dicts to the Question table.

    Each dict expected:
      {
        "type":        "mcq" | "theory" | "coding",
        "difficulty":  "easy" | "medium" | "hard",
        "question":    str,
        "options":     list[str] | None,   # MCQ only
        "answer":      str,
        "explanation": str,
        "topic":       str,
      }

    Returns the number of rows inserted.
    """
    from app import create_app
    from app.extensions import db
    from app.models.question import Question
    from app.models.document import Document

    app = create_app()
    with app.app_context():
        rows = []
        for q in questions:
            row = Question(
                doc_id=doc_id,
                q_type=q.get("type", "mcq"),
                difficulty=q.get("difficulty", "medium"),
                question_text=q["question"],
                options=q.get("options"),       # stored as JSONB
                answer=q["answer"],
                explanation=q.get("explanation", ""),
                topic=q.get("topic", ""),
            )
            rows.append(row)
            db.session.add(row)

        doc = Document.query.get(doc_id)
        if doc:
            doc.question_count = (doc.question_count or 0) + len(rows)

        db.session.commit()
        return len(rows)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="ml.tasks.generate_questions",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
)
def generate_questions_task(
    self,
    doc_id: str,
    q_count: int = _DEFAULT_Q_COUNT,
    difficulty: str = "medium",
    q_types: list = None,
) -> dict:
    """
    Phase B — Question generation pipeline.

    Args:
        doc_id:     UUID of the Document record.
        q_count:    Total number of questions to generate.
        difficulty: "easy" | "medium" | "hard" | "mixed"
        q_types:    List of question types to include, e.g. ["mcq", "theory"].
                    Defaults to ["mcq", "theory", "coding"].

    Returns:
        dict with doc_id, question_count, status.
    """
    if q_types is None:
        q_types = ["mcq", "theory", "coding"]

    logger.info(
        "[qgen] Starting doc_id=%s count=%d difficulty=%s types=%s",
        doc_id, q_count, difficulty, q_types,
    )

    try:
        # ── Step 1: Load document metadata ─────────────────────────────────
        from app import create_app
        from app.models.document import Document

        app = create_app()
        with app.app_context():
            doc = Document.query.get(doc_id)
            if not doc:
                raise ValueError(f"Document {doc_id} not found.")
            subject = doc.subject or "General"
            chunk_count = doc.chunk_count or 0

        if chunk_count == 0:
            raise ValueError(f"No chunks found for doc_id={doc_id}; cannot generate questions.")

        # ── Step 2: RAG — retrieve representative chunks ────────────────────
        store = VectorStore()
        top_k = _CHUNKS_PER_DIFFICULTY.get(difficulty, 7)

        # Use a broad subject query to get diverse coverage
        from ml.pipeline.embedder import Embedder
        embedder = Embedder()
        query_vec = embedder.encode([f"key concepts in {subject}"])[0]

        results = store.query(
            collection_name=f"doc_{doc_id}",
            query_vector=query_vec,
            top_k=min(top_k, chunk_count),
        )
        # results = [{"text": str, "metadata": dict, "score": float}, ...]

        context_chunks = [r["text"] for r in results]
        if not context_chunks:
            raise ValueError("Vector store query returned no chunks.")

        logger.info("[qgen] Retrieved %d context chunks for doc_id=%s", len(context_chunks), doc_id)

        # ── Step 3: Generate questions via LLM ─────────────────────────────
        generator = QuestionGenerator()
        questions = generator.generate(
            context_chunks=context_chunks,
            subject=subject,
            q_count=q_count,
            difficulty=difficulty,
            q_types=q_types,
        )

        if not questions:
            raise ValueError("LLM returned no questions.")

        logger.info("[qgen] Generated %d questions for doc_id=%s", len(questions), doc_id)

        # ── Step 4: Persist to DB ───────────────────────────────────────────
        saved = _save_questions(doc_id, questions)
        logger.info("[qgen] Saved %d questions for doc_id=%s", saved, doc_id)

        return {"doc_id": doc_id, "question_count": saved, "status": "done"}

    except Exception as exc:
        logger.exception("[qgen] Failed for doc_id=%s: %s", doc_id, exc)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))