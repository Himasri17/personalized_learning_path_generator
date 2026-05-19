"""
backend/app/blueprints/analysis/routes.py

REST endpoints for the Analysis blueprint.

POST  /analysis/sessions/<session_id>/grade          trigger grading (sync, dev only)
GET   /analysis/sessions/<session_id>/results        full results + per-question breakdown
GET   /analysis/sessions/<session_id>/summary        lightweight score + VARK card
GET   /analysis/sessions/<session_id>/topics         topic-level accuracy breakdown
GET   /analysis/users/me/stats                       user-level aggregate analytics
GET   /analysis/users/me/history                     paginated session history with scores
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from ...extensions import db
from ...models.quiz_session import QuizSession
from ...models.question     import Question
from ...models.response     import Response
from ...models.vark_profile import VarkProfile
from ...models.user         import User

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _owned_session(session_id: str, user_id: str) -> QuizSession:
    """Return session or abort 404 if not found / not owned by user."""
    session = QuizSession.query.filter_by(id=session_id, user_id=user_id).first()
    if not session:
        from flask import abort
        abort(404, description="Session not found.")
    return session


def _require_completed(session: QuizSession):
    if session.status not in ("completed", "submitted"):
        from flask import abort
        abort(400, description=f"Session is not completed (status={session.status}).")


# ── 1. Trigger grading (synchronous, for development / testing) ───────────────

@analysis_bp.post("/sessions/<session_id>/grade")
@jwt_required()
def trigger_grade(session_id: str):
    """
    Synchronously grades a submitted session.
    In production you'd call score_and_classify.delay(session_id) instead.
    Exposed so the frontend can call it during development without Celery running.
    """
    user_id = get_jwt_identity()
    session = _owned_session(session_id, user_id)

    if session.status == "completed":
        return jsonify({"message": "Already graded.", "session": session.to_dict()}), 200

    if session.status not in ("submitted", "in_progress"):
        return jsonify({"error": f"Cannot grade a session with status '{session.status}'."}), 400

    try:
        from .grader import grade_session
        summary = grade_session(session_id)
    except Exception as exc:
        logger.exception("Synchronous grading failed for session %s", session_id)
        session.status        = "failed"
        session.error_message = str(exc)
        db.session.commit()
        return jsonify({"error": "Grading failed.", "detail": str(exc)}), 500

    # ── VARK classification ───────────────────────────────────────────────────
    vark_profile = None
    try:
        from ...ml.vark.rule_classifier import classify as vark_classify
        v, a, r, k = vark_classify(summary["behaviour_signals"])
        vark_profile = VarkProfile.from_scores(
            user_id=user_id,
            session_id=session_id,
            v=v, a=a, r=r, k=k,
            raw_features=summary["behaviour_signals"],
        )
        db.session.add(vark_profile)
        dominant = vark_profile.dominant_style
    except Exception as exc:
        logger.warning("VARK classification failed: %s", exc)
        dominant = None

    # ── Persist score back to session ─────────────────────────────────────────
    session.mark_completed(
        score        = summary["score"],
        correct      = summary["correct"],
        incorrect    = summary["incorrect"],
        skipped      = summary["skipped"],
        vark_style   = dominant,
        topic_accuracy = summary["topic_accuracy"],
    )
    db.session.commit()

    return jsonify({
        "message": "Grading complete.",
        "session": session.to_dict(),
        "vark":    vark_profile.to_dict() if vark_profile else None,
        "summary": summary,
    }), 200


# ── 2. Full results ────────────────────────────────────────────────────────────

@analysis_bp.get("/sessions/<session_id>/results")
@jwt_required()
def get_results(session_id: str):
    """
    Returns the complete result payload:
      - session metadata + score
      - every question with correct_answer + explanation
      - every response with grading details
      - VARK profile (if available)
    """
    user_id = get_jwt_identity()
    session = _owned_session(session_id, user_id)
    _require_completed(session)

    questions = (
        Question.query
        .filter_by(session_id=session_id, is_active=True)
        .order_by(Question.order_index)
        .all()
    )
    responses = Response.query.filter_by(session_id=session_id).all()
    resp_map  = {r.question_id: r for r in responses}

    vark = (
        VarkProfile.query
        .filter_by(session_id=session_id)
        .order_by(VarkProfile.created_at.desc())
        .first()
    )

    question_details = []
    for q in questions:
        r = resp_map.get(q.id)
        question_details.append({
            "question":  q.to_dict(include_answer=True),
            "response":  r.to_dict(include_feedback=True) if r else None,
        })

    return jsonify({
        "session":          session.to_dict(),
        "question_details": question_details,
        "vark":             vark.to_dict() if vark else None,
    })


# ── 3. Lightweight summary card ───────────────────────────────────────────────

@analysis_bp.get("/sessions/<session_id>/summary")
@jwt_required()
def get_summary(session_id: str):
    """
    Lightweight endpoint for the ResultsPage banner:
      score_percent, correct/incorrect/skipped counts, dominant VARK style.
    """
    user_id = get_jwt_identity()
    session = _owned_session(session_id, user_id)
    _require_completed(session)

    vark = (
        VarkProfile.query
        .filter_by(session_id=session_id)
        .order_by(VarkProfile.created_at.desc())
        .first()
    )

    return jsonify({
        "session_id":     session.id,
        "subject":        session.subject,
        "score":          session.score,
        "score_percent":  session.score_percent,
        "correct_count":  session.correct_count,
        "incorrect_count": session.incorrect_count,
        "skipped_count":  session.skipped_count,
        "total_time_ms":  session.total_time_ms,
        "status":         session.status,
        "vark_style":     session.vark_style,
        "vark_scores":    vark.normalised_scores if vark else None,
        "study_tips":     vark.study_tips if vark else None,
        "completed_at":   session.completed_at.isoformat() if session.completed_at else None,
    })


# ── 4. Topic-level accuracy breakdown ────────────────────────────────────────

@analysis_bp.get("/sessions/<session_id>/topics")
@jwt_required()
def get_topic_breakdown(session_id: str):
    """
    Returns per-topic accuracy so the frontend can render a weakness heatmap.
    Response format:
      [{ "topic": str, "accuracy": float, "question_count": int }]
    """
    user_id = get_jwt_identity()
    session = _owned_session(session_id, user_id)
    _require_completed(session)

    # Use the pre-computed JSON column if available
    if session.topic_accuracy:
        payload = [
            {"topic": t, "accuracy": a, "question_count": None}
            for t, a in session.topic_accuracy.items()
        ]
        return jsonify({"topics": payload})

    # Recompute from raw data
    questions = Question.query.filter_by(session_id=session_id, is_active=True).all()
    responses = Response.query.filter_by(session_id=session_id).all()
    resp_map  = {r.question_id: r for r in responses}

    buckets: dict[str, dict] = {}
    for q in questions:
        r = resp_map.get(q.id)
        awarded  = (r.awarded_score or 0.0) if r else 0.0
        possible = q.max_score
        for tag in (q.topic_tags or ["Uncategorised"]):
            if tag not in buckets:
                buckets[tag] = {"awarded": 0.0, "possible": 0.0, "count": 0}
            buckets[tag]["awarded"]  += awarded
            buckets[tag]["possible"] += possible
            buckets[tag]["count"]    += 1

    payload = [
        {
            "topic":          tag,
            "accuracy":       round(v["awarded"] / v["possible"], 4) if v["possible"] else 0.0,
            "question_count": v["count"],
        }
        for tag, v in sorted(buckets.items())
    ]
    return jsonify({"topics": payload})


# ── 5. User-level aggregate stats ─────────────────────────────────────────────

@analysis_bp.get("/users/me/stats")
@jwt_required()
def get_user_stats():
    """
    Returns aggregate stats across all completed sessions for the current user.
    Used by ProfilePage / dashboard header.
    """
    user_id  = get_jwt_identity()
    user     = User.query.get_or_404(user_id)

    completed = (
        QuizSession.query
        .filter_by(user_id=user_id, status="completed")
        .all()
    )
    all_sessions = QuizSession.query.filter_by(user_id=user_id).count()

    if not completed:
        return jsonify({
            "user_id":           user_id,
            "total_sessions":    all_sessions,
            "completed_sessions": 0,
            "avg_score":         None,
            "best_score":        None,
            "total_questions":   0,
            "dominant_vark":     None,
            "vark_distribution": {},
            "score_trend":       [],
        })

    scores       = [s.score for s in completed if s.score is not None]
    avg_score    = round(sum(scores) / len(scores), 4) if scores else None
    best_score   = round(max(scores), 4) if scores else None
    total_qs     = sum(s.question_count for s in completed)

    # VARK distribution across all sessions
    vark_counts: dict[str, int] = {}
    for s in completed:
        if s.vark_style:
            vark_counts[s.vark_style] = vark_counts.get(s.vark_style, 0) + 1
    dominant_vark = max(vark_counts, key=vark_counts.get) if vark_counts else None

    # Score trend (chronological, for line chart)
    score_trend = [
        {
            "session_id":   s.id,
            "subject":      s.subject,
            "score_percent": s.score_percent,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        }
        for s in sorted(completed, key=lambda x: x.completed_at or datetime.min)
    ]

    return jsonify({
        "user_id":            user_id,
        "full_name":          user.full_name,
        "total_sessions":     all_sessions,
        "completed_sessions": len(completed),
        "avg_score":          avg_score,
        "best_score":         best_score,
        "total_questions":    total_qs,
        "dominant_vark":      dominant_vark,
        "vark_distribution":  vark_counts,
        "score_trend":        score_trend,
    })


# ── 6. Paginated session history ──────────────────────────────────────────────

@analysis_bp.get("/users/me/history")
@jwt_required()
def get_session_history():
    """
    Paginated list of all sessions for the current user.
    Query params:
      page     (int, default 1)
      per_page (int, default 15, max 50)
      status   (optional filter: completed|processing|ready …)
    """
    user_id  = get_jwt_identity()
    page     = max(1, request.args.get("page",     1,  type=int))
    per_page = min(50, request.args.get("per_page", 15, type=int))
    status   = request.args.get("status", None)

    query = QuizSession.query.filter_by(user_id=user_id)
    if status:
        query = query.filter_by(status=status)

    paginated = (
        query
        .order_by(QuizSession.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return jsonify({
        "sessions":   [s.to_dict() for s in paginated.items],
        "total":      paginated.total,
        "page":       page,
        "per_page":   per_page,
        "pages":      paginated.pages,
        "has_next":   paginated.has_next,
        "has_prev":   paginated.has_prev,
    })


# ── 7. Single question re-grade (admin / debug) ───────────────────────────────

@analysis_bp.post("/sessions/<session_id>/responses/<response_id>/regrade")
@jwt_required()
def regrade_response(session_id: str, response_id: str):
    """
    Re-grades a single response. Useful for manual override or debugging.
    Body (optional): { "override_score": float }
    """
    user_id  = get_jwt_identity()
    session  = _owned_session(session_id, user_id)

    resp = Response.query.filter_by(
        id=response_id, session_id=session_id
    ).first_or_404()

    q = resp.question
    data = request.get_json(silent=True) or {}

    # Manual override path
    if "override_score" in data:
        override = float(data["override_score"])
        if not (0.0 <= override <= q.max_score):
            return jsonify({"error": f"override_score must be between 0 and {q.max_score}."}), 400
        resp.mark_graded(
            is_correct=override >= q.max_score,
            awarded_score=override,
            feedback=f"Manually overridden to {override}.",
        )
        db.session.commit()
        return jsonify({"response": resp.to_dict()})

    # Automatic re-grade
    from .grader import grade_mcq, grade_theory, grade_coding
    if q.question_type == "mcq":
        result = grade_mcq(resp.user_answer, q.correct_answer, q.max_score)
    elif q.question_type == "theory":
        result = grade_theory(resp.user_answer, q.correct_answer, q.max_score)
    else:
        result = grade_coding(resp.user_answer, q.correct_answer, q.test_cases, q.language, q.max_score)

    resp.mark_graded(
        is_correct=result.is_correct,
        awarded_score=result.awarded_score,
        similarity_score=result.similarity_score,
        test_results=result.test_results,
        feedback=result.feedback,
    )
    db.session.commit()
    return jsonify({"response": resp.to_dict()})