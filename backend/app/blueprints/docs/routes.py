"""
backend/app/blueprints/docs/routes.py

Blueprint: docs
Endpoints:
  POST /docs/upload        — Upload a PDF/document; enqueue ML ingestion pipeline
  GET  /docs/<doc_id>/status — Poll document processing status
"""

import uuid
import os
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from app.extensions import db
from app.blueprints.docs.storage import upload_to_storage, allowed_file, get_file_url
from ml.tasks.ingest_task import ingest_document_task

docs_bp = Blueprint("docs", __name__, url_prefix="/docs")

# ---------------------------------------------------------------------------
# Allowed MIME types / extensions
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
MAX_FILE_SIZE_MB = 20


# ---------------------------------------------------------------------------
# POST /docs/upload
# ---------------------------------------------------------------------------
@docs_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_document():
    """
    Upload a study document (PDF, DOCX, TXT, MD).

    Form-data fields:
      file        (required) – binary file
      subject     (optional) – e.g. "Machine Learning"
      description (optional) – short note

    Returns 202 Accepted with a doc_id the client can poll.
    """
    current_user_id = get_jwt_identity()

    # ── Validate file presence ──────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # ── Validate file size (stream-safe check) ──────────────────────────────
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()
    file.seek(0)

    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"File exceeds {MAX_FILE_SIZE_MB} MB limit."}), 413

    # ── Generate unique document record ────────────────────────────────────
    doc_id = str(uuid.uuid4())
    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit(".", 1)[-1].lower()
    storage_key = f"documents/{current_user_id}/{doc_id}.{extension}"

    # ── Upload raw file to S3 / MinIO ───────────────────────────────────────
    try:
        upload_to_storage(
            file_obj=file,
            key=storage_key,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as exc:
        current_app.logger.error("Storage upload failed: %s", exc)
        return jsonify({"error": "File storage failed. Please try again."}), 500

    # ── Persist document metadata to DB ────────────────────────────────────
    from app.models.document import Document  # local import to avoid circular deps

    doc = Document(
        id=doc_id,
        user_id=current_user_id,
        original_filename=original_filename,
        storage_key=storage_key,
        subject=request.form.get("subject", ""),
        description=request.form.get("description", ""),
        status="queued",
    )
    db.session.add(doc)
    db.session.commit()

    # ── Enqueue Celery ML pipeline (Phase A: ingest → embed → FAISS) ───────
    ingest_document_task.apply_async(
        args=[doc_id, storage_key],
        task_id=f"ingest-{doc_id}",
    )

    return jsonify({
        "message": "Document uploaded successfully. Processing has started.",
        "doc_id": doc_id,
        "status": "queued",
        "filename": original_filename,
    }), 202


# ---------------------------------------------------------------------------
# GET /docs/<doc_id>/status
# ---------------------------------------------------------------------------
@docs_bp.route("/<doc_id>/status", methods=["GET"])
@jwt_required()
def get_document_status(doc_id):
    """
    Poll the processing status of an uploaded document.

    Status lifecycle:
      queued → processing → ready | failed

    Returns document metadata + current status.
    """
    current_user_id = get_jwt_identity()

    from app.models.document import Document

    doc = Document.query.filter_by(id=doc_id, user_id=current_user_id).first()

    if not doc:
        return jsonify({"error": "Document not found."}), 404

    payload = {
        "doc_id": doc.id,
        "filename": doc.original_filename,
        "subject": doc.subject,
        "status": doc.status,           # queued | processing | ready | failed
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
    }

    # Expose the download URL only once processing is complete
    if doc.status == "ready":
        payload["file_url"] = get_file_url(doc.storage_key)
        payload["chunk_count"] = doc.chunk_count       # how many FAISS vectors stored
        payload["question_count"] = doc.question_count # Qs generated (Phase B)

    if doc.status == "failed":
        payload["error_message"] = doc.error_message

    return jsonify(payload), 200


# ---------------------------------------------------------------------------
# GET /docs/  (list user's documents)
# ---------------------------------------------------------------------------
@docs_bp.route("/", methods=["GET"])
@jwt_required()
def list_documents():
    """
    Return all documents uploaded by the authenticated user.

    Query params:
      status  (optional) – filter by status (e.g. ?status=ready)
      subject (optional) – filter by subject
    """
    current_user_id = get_jwt_identity()

    from app.models.document import Document

    query = Document.query.filter_by(user_id=current_user_id)

    status_filter = request.args.get("status")
    if status_filter:
        query = query.filter_by(status=status_filter)

    subject_filter = request.args.get("subject")
    if subject_filter:
        query = query.filter(Document.subject.ilike(f"%{subject_filter}%"))

    docs = query.order_by(Document.created_at.desc()).all()

    return jsonify({
        "documents": [
            {
                "doc_id": d.id,
                "filename": d.original_filename,
                "subject": d.subject,
                "status": d.status,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "total": len(docs),
    }), 200


# ---------------------------------------------------------------------------
# DELETE /docs/<doc_id>
# ---------------------------------------------------------------------------
@docs_bp.route("/<doc_id>", methods=["DELETE"])
@jwt_required()
def delete_document(doc_id):
    """
    Soft-delete a document record. The S3 object is NOT removed here
    (handled by a scheduled cleanup job to avoid data loss on retries).
    """
    current_user_id = get_jwt_identity()

    from app.models.document import Document

    doc = Document.query.filter_by(id=doc_id, user_id=current_user_id).first()
    if not doc:
        return jsonify({"error": "Document not found."}), 404

    doc.is_deleted = True
    db.session.commit()

    return jsonify({"message": "Document deleted."}), 200