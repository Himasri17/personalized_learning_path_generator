"""
backend/app/blueprints/analysis/grader.py

Grades user responses for all three question types:
  - MCQ     → exact string match
  - Theory  → cosine similarity between sentence-transformer embeddings
  - Coding  → runs user code in a sandboxed subprocess and checks test cases

All public functions return a GradeResult dataclass.
The Celery task (score_and_classify) calls grade_session() which iterates
every Response in a session, calls the right grader, and persists results.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session as DBSession

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    is_correct: bool | None       # None = partial (theory with mid similarity)
    awarded_score: float          # 0.0 – question.max_score
    similarity_score: float | None = None
    test_results: list[dict] | None = None
    feedback: str | None = None
    error: str | None = None


# ── MCQ grader ───────────────────────────────────────────────────────────────

def grade_mcq(user_answer: str | None, correct_answer: str, max_score: float = 1.0) -> GradeResult:
    """
    Exact match after normalising whitespace and casing.
    Accepts the full option text OR just the letter (A/B/C/D).
    """
    if not user_answer:
        return GradeResult(is_correct=False, awarded_score=0.0, feedback="No answer provided.")

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    ua = _norm(user_answer)
    ca = _norm(correct_answer)

    # Direct text match
    if ua == ca:
        return GradeResult(is_correct=True, awarded_score=max_score, feedback="Correct!")

    # Letter-only match: user sent "A" and correct_answer starts with "A."
    if len(ua) == 1 and ca.startswith(ua + "."):
        return GradeResult(is_correct=True, awarded_score=max_score, feedback="Correct!")

    return GradeResult(
        is_correct=False,
        awarded_score=0.0,
        feedback=f"Incorrect. The correct answer was: {correct_answer}",
    )


# ── Theory grader ─────────────────────────────────────────────────────────────

# Lazy-load the embedding model so workers that only grade MCQs don't pay the
# ~400 MB load cost.
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformer model for theory grading.")
        except Exception as exc:
            logger.error("Failed to load sentence-transformer: %s", exc)
            raise
    return _embedder


def _cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Thresholds — tune these after evaluation
THEORY_CORRECT_THRESHOLD  = 0.80   # ≥ 0.80  → fully correct
THEORY_PARTIAL_THRESHOLD  = 0.55   # 0.55–0.80 → partial credit (50 %)
# < 0.55 → incorrect


def grade_theory(
    user_answer: str | None,
    correct_answer: str,
    max_score: float = 1.0,
) -> GradeResult:
    """
    Embeds both answers with sentence-transformers and computes cosine similarity.
    Returns full / partial / zero credit based on similarity thresholds.
    """
    if not user_answer or not user_answer.strip():
        return GradeResult(
            is_correct=False, awarded_score=0.0,
            similarity_score=0.0, feedback="No answer provided.",
        )

    try:
        model = _get_embedder()
        embeddings = model.encode([user_answer.strip(), correct_answer.strip()])
        sim = _cosine_similarity(embeddings[0], embeddings[1])
    except Exception as exc:
        logger.exception("Theory grader embedding failed: %s", exc)
        return GradeResult(
            is_correct=None, awarded_score=0.0,
            error=f"Grading error: {exc}",
        )

    sim = round(sim, 4)

    if sim >= THEORY_CORRECT_THRESHOLD:
        return GradeResult(
            is_correct=True,
            awarded_score=max_score,
            similarity_score=sim,
            feedback=f"Excellent answer! (similarity {sim:.0%})",
        )
    elif sim >= THEORY_PARTIAL_THRESHOLD:
        partial = round(max_score * 0.5, 4)
        return GradeResult(
            is_correct=None,          # partial — not strictly True or False
            awarded_score=partial,
            similarity_score=sim,
            feedback=(
                f"Partially correct (similarity {sim:.0%}). "
                f"Model answer: {correct_answer[:300]}"
            ),
        )
    else:
        return GradeResult(
            is_correct=False,
            awarded_score=0.0,
            similarity_score=sim,
            feedback=(
                f"Incorrect (similarity {sim:.0%}). "
                f"Model answer: {correct_answer[:300]}"
            ),
        )


# ── Coding grader ─────────────────────────────────────────────────────────────

# Languages we can sandbox-execute locally (extend as needed)
SUPPORTED_LANGUAGES = {"python", "python3"}

# Hard limits for the subprocess sandbox
SANDBOX_TIMEOUT_SECONDS = 5
SANDBOX_MAX_OUTPUT_BYTES = 8_192     # 8 KB stdout/stderr cap


def _run_python_code(code: str, stdin_data: str = "") -> tuple[str, str, int]:
    """
    Execute *code* in a child Python process with tight resource limits.
    Returns (stdout, stderr, returncode).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT_SECONDS,
        )
        stdout = result.stdout[:SANDBOX_MAX_OUTPUT_BYTES]
        stderr = result.stderr[:SANDBOX_MAX_OUTPUT_BYTES]
        return stdout, stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TimeoutError: execution exceeded 5 seconds.", 1
    except Exception as exc:
        return "", str(exc), 1


def _build_test_harness(user_code: str, test_case: dict) -> str:
    """
    Wrap user code + one test case into a single Python script.
    test_case format: {"input": "...", "expected_output": "..."}
    The harness prints PASS or FAIL:<actual>.
    """
    input_val = json.dumps(test_case.get("input", ""))
    expected  = str(test_case.get("expected_output", "")).strip()

    harness = textwrap.dedent(f"""
import sys, json, io

_input_data = json.loads({input_val!r})

# Redirect stdin so code can use input()
sys.stdin = io.StringIO(str(_input_data) if not isinstance(_input_data, str) else _input_data)

# ---- user code ----
{user_code}
# ---- end user code ----
""")
    # The harness itself can't call the function — we need the LLM to include
    # a `main()` or a direct call. For simplicity we capture stdout.
    return harness


def grade_coding(
    user_answer: str | None,
    correct_answer: str,
    test_cases: list[dict] | None,
    language: str | None = "python",
    max_score: float = 1.0,
) -> GradeResult:
    """
    Runs user code against each test case in a subprocess sandbox.
    Falls back to cosine-similarity against the reference solution when the
    language is unsupported or test_cases is empty.
    """
    if not user_answer or not user_answer.strip():
        return GradeResult(
            is_correct=False, awarded_score=0.0,
            feedback="No code submitted.",
        )

    lang = (language or "python").lower()

    # Fallback: unsupported language or no test cases → similarity grade
    if lang not in SUPPORTED_LANGUAGES or not test_cases:
        logger.warning(
            "Coding grader falling back to similarity for language=%s, "
            "test_cases=%s", lang, bool(test_cases)
        )
        return grade_theory(user_answer, correct_answer, max_score)

    results: list[dict] = []
    passed = 0

    for i, tc in enumerate(test_cases):
        harness = _build_test_harness(user_answer, tc)
        stdout, stderr, rc = _run_python_code(harness)

        expected = str(tc.get("expected_output", "")).strip()
        actual   = stdout.strip()
        ok       = (actual == expected) and rc == 0

        if ok:
            passed += 1

        results.append({
            "test_index":      i,
            "input":           tc.get("input"),
            "expected_output": expected,
            "actual_output":   actual,
            "stderr":          stderr[:500] if stderr else None,
            "passed":          ok,
        })

    total = len(test_cases)
    ratio = passed / total if total else 0.0
    awarded = round(max_score * ratio, 4)
    is_correct = passed == total

    feedback = (
        f"{passed}/{total} test cases passed."
        if not is_correct
        else f"All {total} test cases passed!"
    )

    return GradeResult(
        is_correct=is_correct,
        awarded_score=awarded,
        test_results=results,
        feedback=feedback,
    )


# ── Session-level grading (called by Celery task) ────────────────────────────

def grade_session(session_id: str) -> dict:
    """
    Grades all ungraded responses in a QuizSession.
    Writes results to DB and returns a summary dict consumed by score_and_classify.

    Returns:
        {
          "total": int,
          "correct": int,
          "incorrect": int,
          "skipped": int,
          "score": float,           # 0.0 – 1.0 (awarded / max possible)
          "topic_accuracy": dict,   # {topic_tag: accuracy_float}
          "behaviour_signals": dict # aggregated for VARK classifier
        }
    """
    from ...models.quiz_session import QuizSession
    from ...models.question     import Question
    from ...models.response     import Response
    from ...extensions          import db

    session   = QuizSession.query.get(session_id)
    if not session:
        raise ValueError(f"QuizSession {session_id!r} not found.")

    responses = Response.query.filter_by(session_id=session_id).all()
    if not responses:
        logger.warning("grade_session: no responses for session %s", session_id)
        return _empty_summary()

    total_awarded = 0.0
    total_possible = 0.0
    correct = incorrect = skipped_count = 0

    # topic → [awarded, possible]
    topic_buckets: dict[str, list[float]] = {}

    # behaviour aggregation
    time_values:          list[int]   = []
    hint_count            = 0
    answer_change_count   = 0
    tab_switch_total      = 0
    fast_answer_count     = 0   # answered in < 10 000 ms

    for resp in responses:
        q: Question = resp.question

        if resp.skipped:
            resp.mark_skipped()
            skipped_count += 1
            total_possible += q.max_score
            _update_topic_buckets(topic_buckets, q.topic_tags, 0.0, q.max_score)
            continue

        # ── Grade ─────────────────────────────────────────────────────────
        if q.question_type == "mcq":
            result = grade_mcq(resp.user_answer, q.correct_answer, q.max_score)
        elif q.question_type == "theory":
            result = grade_theory(resp.user_answer, q.correct_answer, q.max_score)
        else:  # coding
            result = grade_coding(
                resp.user_answer, q.correct_answer,
                q.test_cases, q.language, q.max_score,
            )

        if result.error:
            resp.grading_status = "errored"
            resp.grader_feedback = result.error
            db.session.add(resp)
            total_possible += q.max_score
            continue

        resp.mark_graded(
            is_correct=result.is_correct,
            awarded_score=result.awarded_score,
            similarity_score=result.similarity_score,
            test_results=result.test_results,
            feedback=result.feedback,
        )
        db.session.add(resp)

        total_awarded  += result.awarded_score
        total_possible += q.max_score
        _update_topic_buckets(topic_buckets, q.topic_tags, result.awarded_score, q.max_score)

        if result.is_correct is True:
            correct += 1
        elif result.is_correct is False:
            incorrect += 1
        # None (partial theory) → counted in neither bucket

        # ── Behaviour signals ──────────────────────────────────────────────
        if resp.time_taken_ms is not None:
            time_values.append(resp.time_taken_ms)
            if resp.time_taken_ms < 10_000:
                fast_answer_count += 1
        if resp.hint_used:
            hint_count += 1
        if resp.answer_changed:
            answer_change_count += 1
        tab_switch_total += resp.tab_switches or 0

    db.session.commit()

    score     = round(total_awarded / total_possible, 4) if total_possible else 0.0
    n         = len(responses)
    avg_time  = int(sum(time_values) / len(time_values)) if time_values else 0

    topic_accuracy = {
        tag: round(vals[0] / vals[1], 4) if vals[1] else 0.0
        for tag, vals in topic_buckets.items()
    }

    behaviour_signals = {
        "mcq_accuracy":        _type_accuracy(responses, "mcq"),
        "theory_accuracy":     _type_accuracy(responses, "theory"),
        "coding_accuracy":     _type_accuracy(responses, "coding"),
        "avg_time_ms":         avg_time,
        "skip_ratio":          round(skipped_count / n, 4) if n else 0.0,
        "fast_answer_ratio":   round(fast_answer_count / n, 4) if n else 0.0,
        "hint_used_ratio":     round(hint_count / n, 4) if n else 0.0,
        "answer_change_ratio": round(answer_change_count / n, 4) if n else 0.0,
        "tab_switch_ratio":    round(tab_switch_total / n, 4) if n else 0.0,
        "total_questions":     n,
    }

    return {
        "total":              n,
        "correct":            correct,
        "incorrect":          incorrect,
        "skipped":            skipped_count,
        "score":              score,
        "topic_accuracy":     topic_accuracy,
        "behaviour_signals":  behaviour_signals,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty_summary() -> dict:
    return {
        "total": 0, "correct": 0, "incorrect": 0, "skipped": 0,
        "score": 0.0, "topic_accuracy": {}, "behaviour_signals": {},
    }


def _update_topic_buckets(
    buckets: dict[str, list[float]],
    tags: list[str] | None,
    awarded: float,
    possible: float,
) -> None:
    for tag in (tags or []):
        if tag not in buckets:
            buckets[tag] = [0.0, 0.0]
        buckets[tag][0] += awarded
        buckets[tag][1] += possible


def _type_accuracy(responses: list, qtype: str) -> float:
    """Return accuracy (awarded/possible) for one question_type across responses."""
    subset = [r for r in responses if r.question and r.question.question_type == qtype]
    if not subset:
        return 0.0
    awarded  = sum(r.awarded_score or 0.0 for r in subset)
    possible = sum(r.question.max_score for r in subset)
    return round(awarded / possible, 4) if possible else 0.0