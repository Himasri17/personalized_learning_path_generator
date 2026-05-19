"""
quiz/routes.py
--------------
Blueprint: /sessions
POST   /sessions                        – create a new quiz session
GET    /sessions/<session_id>/questions – fetch paginated questions
POST   /sessions/<session_id>/submit    – submit answers, trigger grading
GET    /sessions/<session_id>/status    – poll session / grading status
DELETE /sessions/<session_id>           – abandon an in-progress session
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, validate, ValidationError

from app.extensions import db, celery
from app.models.quiz_session import QuizSession
from app.models.question import Question
from app.models.response import Response as QuizResponse
from app.models.vark_profile import VarkProfile
from ml.tasks.qgen_task import generate_questions_task
from .sandbox import run_code_in_sandbox
from analysis.grader import grade_session          # relative import within backend

logger = logging.getLogger(__name__)

quiz_bp = Blueprint("quiz", __name__, url_prefix="/sessions")

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class SessionCreateSchema(Schema):
    subject      = fields.Str(required=True, validate=validate.Length(min=2, max=120))
    difficulty   = fields.Str(
        load_default="intermediate",
        validate=validate.OneOf(["beginner", "intermediate", "advanced"]),
    )
    question_count = fields.Int(
        load_default=10,
        validate=validate.Range(min=3, max=50),
    )
    question_types = fields.List(
        fields.Str(validate=validate.OneOf(["mcq", "theory", "coding"])),
        load_default=["mcq"],
    )
    doc_id = fields.Str(load_default=None)   # optional uploaded document UUID


class AnswerSchema(Schema):
    question_id     = fields.Str(required=True)
    selected_answer = fields.Str(load_default=None)   # MCQ / theory
    code_answer     = fields.Str(load_default=None)   # coding questions
    time_taken_ms   = fields.Int(load_default=0)      # milliseconds on this question


class SubmitSchema(Schema):
    answers   = fields.List(fields.Nested(AnswerSchema), required=True, validate=validate.Length(min=1))
    completed = fields.Bool(load_default=True)        # False = partial save


_session_schema = SessionCreateSchema()
_submit_schema  = SubmitSchema()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_session_or_404(session_id: str, user_id: str) -> QuizSession:
    session = QuizSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        return None
    return session


def _serialize_question(q: Question, include_answer: bool = False) -> dict:
    """Serialize a Question ORM object to a plain dict for JSON output."""
    data = {
        "id":         str(q.id),
        "text":       q.text,
        "q_type":     q.q_type,
        "topic":      q.topic,
        "difficulty": q.difficulty,
        "options":    q.options,       # list[str] for MCQ; None otherwise
        "code_stub":  q.code_stub,     # starter code for coding Qs
        "test_cases": q.test_cases if q.q_type == "coding" else None,
    }
    if include_answer:
        data["correct_answer"] = q.correct_answer
    return data


# ---------------------------------------------------------------------------
# POST /sessions  – Create session & kick off async question generation
# ---------------------------------------------------------------------------

@quiz_bp.post("")
@jwt_required()
def create_session():
    """
    Create a new quiz session.

    Request body (JSON):
        subject        str        required
        difficulty     str        beginner | intermediate | advanced
        question_count int        3-50  (default 10)
        question_types list[str]  mcq | theory | coding
        doc_id         str | null UUID of a previously uploaded document

    Response 202:
        {
          "session_id": "...",
          "status":     "generating",
          "poll_url":   "/sessions/<id>/status"
        }
    """
    user_id = get_jwt_identity()

    try:
        data = _session_schema.load(request.get_json(force=True) or {})
    except ValidationError as err:
        return jsonify({"errors": err.messages}), 422

    session_id = str(uuid.uuid4())
    session = QuizSession(
        id             = session_id,
        user_id        = user_id,
        subject        = data["subject"],
        difficulty     = data["difficulty"],
        question_count = data["question_count"],
        question_types = data["question_types"],
        doc_id         = data["doc_id"],
        status         = "generating",
        created_at     = _utcnow(),
    )
    db.session.add(session)
    db.session.commit()

    # Fire off Celery task to run RAG → LLM question generation
    generate_questions_task.apply_async(
        kwargs={
            "session_id":     session_id,
            "subject":        data["subject"],
            "difficulty":     data["difficulty"],
            "question_count": data["question_count"],
            "question_types": data["question_types"],
            "doc_id":         data["doc_id"],
        },
        countdown=0,
    )

    logger.info("Session %s created for user %s (async qgen dispatched)", session_id, user_id)

    return jsonify(
        {
            "session_id": session_id,
            "status":     "generating",
            "poll_url":   f"/sessions/{session_id}/status",
        }
    ), 202


# ---------------------------------------------------------------------------
# GET /sessions/<session_id>/status  – Poll generation / grading progress
# ---------------------------------------------------------------------------

@quiz_bp.get("/<session_id>/status")
@jwt_required()
def get_status(session_id: str):
    """
    Poll session status.

    Response 200:
        {
          "session_id":      "...",
          "status":          "generating" | "ready" | "in_progress"
                              | "grading" | "completed" | "error",
          "question_count":  10,
          "questions_ready": 10,   // how many Qs have been stored so far
          "score":           null | float,
          "error_message":   null | str
        }
    """
    user_id = get_jwt_identity()
    session = _get_session_or_404(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    questions_ready = Question.query.filter_by(session_id=session_id).count()

    return jsonify(
        {
            "session_id":      session_id,
            "status":          session.status,
            "question_count":  session.question_count,
            "questions_ready": questions_ready,
            "score":           session.score,
            "error_message":   session.error_message,
        }
    ), 200


# ---------------------------------------------------------------------------
# GET /sessions/<session_id>/questions  – Fetch questions (paginated)
# ---------------------------------------------------------------------------

@quiz_bp.get("/<session_id>/questions")
@jwt_required()
def get_questions(session_id: str):
    """
    Fetch quiz questions.  Only available once status == 'ready' or 'in_progress'.

    Query params:
        page  int  (default 1)
        per_page int  (default 10, max 50)

    Response 200:
        {
          "session_id": "...",
          "page": 1,
          "total": 10,
          "questions": [ { id, text, q_type, topic, difficulty, options,
                           code_stub, test_cases } ]
        }
    """
    user_id = get_jwt_identity()
    session = _get_session_or_404(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if session.status == "generating":
        return jsonify({"error": "Questions are still being generated.", "status": "generating"}), 202

    if session.status == "error":
        return jsonify({"error": session.error_message or "Question generation failed."}), 500

    # Mark as in_progress on first fetch
    if session.status == "ready":
        session.status     = "in_progress"
        session.started_at = _utcnow()
        db.session.commit()

    page     = max(1, request.args.get("page", 1, type=int))
    per_page = min(50, max(1, request.args.get("per_page", 10, type=int)))

    pagination = (
        Question.query
        .filter_by(session_id=session_id)
        .order_by(Question.position)
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return jsonify(
        {
            "session_id": session_id,
            "page":       pagination.page,
            "pages":      pagination.pages,
            "total":      pagination.total,
            "questions":  [_serialize_question(q) for q in pagination.items],
        }
    ), 200


# ---------------------------------------------------------------------------
# POST /sessions/<session_id>/submit  – Submit answers + trigger grading
# ---------------------------------------------------------------------------

@quiz_bp.post("/<session_id>/submit")
@jwt_required()
def submit_answers(session_id: str):
    """
    Submit one or more answers.  Call multiple times for partial saves, or once
    with completed=true (default) to finalise and trigger grading.

    Request body:
        {
          "answers": [
            {
              "question_id":     "uuid",
              "selected_answer": "B",           // MCQ or theory
              "code_answer":     "def foo(): …", // coding
              "time_taken_ms":   4200
            }
          ],
          "completed": true
        }

    Response 200 (partial save):
        { "saved": 5, "completed": false }

    Response 202 (grading triggered):
        { "saved": 10, "completed": true, "grade_poll_url": "/sessions/<id>/status" }
    """
    user_id = get_jwt_identity()
    session = _get_session_or_404(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if session.status not in ("in_progress", "ready"):
        return jsonify({"error": f"Cannot submit to a session with status '{session.status}'."}), 400

    try:
        data = _submit_schema.load(request.get_json(force=True) or {})
    except ValidationError as err:
        return jsonify({"errors": err.messages}), 422

    answers   = data["answers"]
    completed = data["completed"]

    # Validate question ownership
    question_ids = [a["question_id"] for a in answers]
    valid_qs = {
        str(q.id): q
        for q in Question.query.filter(
            Question.session_id == session_id,
            Question.id.in_(question_ids),
        ).all()
    }

    saved_count = 0
    for ans in answers:
        qid = ans["question_id"]
        if qid not in valid_qs:
            logger.warning("Unknown question_id %s for session %s — skipping", qid, session_id)
            continue

        question = valid_qs[qid]

        # Handle coding questions: run in sandbox to get stdout / pass-fail
        code_output      = None
        sandbox_passed   = None
        sandbox_error    = None

        if question.q_type == "coding" and ans.get("code_answer"):
            result = run_code_in_sandbox(
                code        = ans["code_answer"],
                language    = question.language or "python",
                test_cases  = question.test_cases or [],
                timeout_sec = 10,
            )
            code_output    = result.get("stdout", "")
            sandbox_passed = result.get("all_passed", False)
            sandbox_error  = result.get("error")

        # Upsert response row
        existing = QuizResponse.query.filter_by(
            session_id  = session_id,
            question_id = qid,
        ).first()

        if existing:
            existing.selected_answer = ans.get("selected_answer")
            existing.code_answer     = ans.get("code_answer")
            existing.code_output     = code_output
            existing.sandbox_passed  = sandbox_passed
            existing.sandbox_error   = sandbox_error
            existing.time_taken_ms   = ans.get("time_taken_ms", 0)
            existing.submitted_at    = _utcnow()
        else:
            resp = QuizResponse(
                id              = str(uuid.uuid4()),
                session_id      = session_id,
                question_id     = qid,
                user_id         = user_id,
                selected_answer = ans.get("selected_answer"),
                code_answer     = ans.get("code_answer"),
                code_output     = code_output,
                sandbox_passed  = sandbox_passed,
                sandbox_error   = sandbox_error,
                time_taken_ms   = ans.get("time_taken_ms", 0),
                submitted_at    = _utcnow(),
            )
            db.session.add(resp)

        saved_count += 1

    db.session.commit()

    if not completed:
        return jsonify({"saved": saved_count, "completed": False}), 200

    # ---- Finalise session and trigger async grading -------------------------
    session.status       = "grading"
    session.completed_at = _utcnow()
    db.session.commit()

    # Grading runs in a Celery worker (MCQ exact-match, theory cosine-sim,
    # coding unit tests) and writes back score + VARK profile
    from ml.tasks.grade_task import grade_session_task
    grade_session_task.apply_async(
        kwargs={"session_id": session_id, "user_id": user_id},
        countdown=0,
    )

    logger.info("Session %s submitted by user %s — grading dispatched", session_id, user_id)

    return jsonify(
        {
            "saved":          saved_count,
            "completed":      True,
            "grade_poll_url": f"/sessions/{session_id}/status",
        }
    ), 202


# ---------------------------------------------------------------------------
# DELETE /sessions/<session_id>  – Abandon an in-progress session
# ---------------------------------------------------------------------------

@quiz_bp.delete("/<session_id>")
@jwt_required()
def abandon_session(session_id: str):
    """
    Mark a session as abandoned.  Idempotent: safe to call multiple times.

    Response 200: { "session_id": "...", "status": "abandoned" }
    """
    user_id = get_jwt_identity()
    session = _get_session_or_404(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if session.status in ("completed", "grading"):
        return jsonify({"error": "Cannot abandon a completed or grading session."}), 400

    session.status = "abandoned"
    db.session.commit()

    return jsonify({"session_id": session_id, "status": "abandoned"}), 200