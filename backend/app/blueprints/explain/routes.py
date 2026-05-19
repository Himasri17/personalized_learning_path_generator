"""
explain/routes.py
-----------------
Blueprint: /explain
POST /explain/<session_id>  →  Server-Sent Events stream of adaptive explanation
"""

import json
import logging
from flask import Blueprint, Response, request, stream_with_context, g
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.extensions import db
from app.models.quiz_session import QuizSession
from app.models.vark_profile import VarkProfile
from app.models.response import Response as QuizResponse
from app.models.question import Question
from .prompt_templates import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

explain_bp = Blueprint("explain", __name__, url_prefix="/explain")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_or_404(session_id: str, user_id: str) -> QuizSession:
    """Return the session if it belongs to the requesting user, else raise 404."""
    session = QuizSession.query.filter_by(id=session_id, user_id=user_id).first()
    if session is None:
        raise ValueError(f"Session {session_id} not found for user {user_id}")
    return session


def _load_wrong_answers(session_id: str) -> list[dict]:
    """
    Return a list of dicts describing every question the user got wrong.
    Each dict has: question_text, correct_answer, user_answer, topic, q_type.
    """
    rows = (
        db.session.query(Question, QuizResponse)
        .join(QuizResponse, Question.id == QuizResponse.question_id)
        .filter(
            QuizResponse.session_id == session_id,
            QuizResponse.is_correct == False,          # noqa: E712
        )
        .all()
    )

    wrong = []
    for question, resp in rows:
        wrong.append(
            {
                "question_text": question.text,
                "correct_answer": question.correct_answer,
                "user_answer": resp.selected_answer,
                "topic": question.topic or "General",
                "q_type": question.q_type,  # mcq | theory | coding
            }
        )
    return wrong


def _get_vark_profile(user_id: str) -> dict:
    """Return the latest VARK profile for a user as a plain dict."""
    profile = (
        VarkProfile.query.filter_by(user_id=user_id)
        .order_by(VarkProfile.created_at.desc())
        .first()
    )
    if profile is None:
        return {"visual": 25, "auditory": 25, "reading": 25, "kinesthetic": 25}
    return {
        "visual": profile.visual,
        "auditory": profile.auditory,
        "reading": profile.reading,
        "kinesthetic": profile.kinesthetic,
        "dominant": profile.dominant_style,
    }


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

def _sse_event(data: str, event: str = "message") -> str:
    """Format a single SSE frame."""
    lines = "\n".join(f"data: {line}" for line in data.splitlines())
    return f"event: {event}\n{lines}\n\n"


def _sse_error(message: str) -> str:
    payload = json.dumps({"error": message})
    return _sse_event(payload, event="error")


def _stream_explanation(
    vark: dict,
    wrong_answers: list[dict],
    subject: str,
    difficulty: str,
) -> "Generator[str, None, None]":
    """
    Core streaming generator.  Calls the LLM with a VARK-adapted system prompt
    and yields SSE frames as tokens arrive.

    Swap the body of this function for your actual LLM SDK (OpenAI, Anthropic,
    Ollama, etc.).  The surrounding SSE plumbing stays the same.
    """
    import openai  # or: from anthropic import Anthropic

    system_prompt = build_system_prompt(vark)
    user_prompt = build_user_prompt(wrong_answers, subject, difficulty)

    client = openai.OpenAI()  # picks up OPENAI_API_KEY from env

    # ---- yield a "start" frame so the client knows streaming began ----------
    yield _sse_event(json.dumps({"status": "start"}), event="start")

    try:
        with client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.7,
            max_tokens=2048,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    payload = json.dumps({"token": delta.content})
                    yield _sse_event(payload)

    except openai.OpenAIError as exc:
        logger.exception("LLM streaming error: %s", exc)
        yield _sse_error(str(exc))
        return

    # ---- terminal "done" frame ----------------------------------------------
    yield _sse_event(json.dumps({"status": "done"}), event="done")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@explain_bp.post("/<session_id>")
@jwt_required()
def stream_explanation(session_id: str):
    """
    POST /explain/<session_id>

    Optional JSON body:
        {
            "focus_topics": ["Recursion", "Big-O"]   // filter wrong answers
        }

    Returns: text/event-stream  (SSE)

    SSE event types:
        start   – {"status": "start"}
        message – {"token": "<partial text>"}
        error   – {"error": "<message>"}
        done    – {"status": "done"}
    """
    user_id = get_jwt_identity()

    # --- validate session ownership -----------------------------------------
    try:
        session = _get_session_or_404(session_id, user_id)
    except ValueError as exc:
        return {"error": str(exc)}, 404

    if session.status != "completed":
        return {"error": "Session is not yet completed."}, 400

    # --- optional topic filter from request body ----------------------------
    body = request.get_json(silent=True) or {}
    focus_topics: list[str] = body.get("focus_topics", [])

    # --- load data ----------------------------------------------------------
    wrong_answers = _load_wrong_answers(session_id)

    if focus_topics:
        wrong_answers = [
            w for w in wrong_answers if w["topic"] in focus_topics
        ]

    if not wrong_answers:
        return {
            "error": "No incorrect answers to explain — great score!"
        }, 200

    vark = _get_vark_profile(user_id)

    # --- build streaming response -------------------------------------------
    generator = _stream_explanation(
        vark=vark,
        wrong_answers=wrong_answers,
        subject=session.subject,
        difficulty=session.difficulty,
    )

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",        # tell Nginx: don't buffer SSE
        "Content-Type": "text/event-stream; charset=utf-8",
    }

    return Response(
        stream_with_context(generator),
        status=200,
        headers=headers,
        mimetype="text/event-stream",
        direct_passthrough=True,
    )