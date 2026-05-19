"""
ml/pipeline/question_generator.py
==================================
Builds the LLM prompt from RAG-retrieved chunks, calls OpenAI / Gemini,
parses the structured JSON response, and returns a validated list of
question dicts ready to be stored in PostgreSQL.

Flow
----
  vector_store  ──► retrieve_chunks(query, k)
                         │
                         ▼
              _build_prompt(chunks, config)
                         │
                         ▼
              _call_llm(prompt)          ← OpenAI GPT-4o or Gemini
                         │
                         ▼
              _parse_response(raw_json)
                         │
                         ▼
              [validated QuestionDict list]
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ml.pipeline.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

QuestionType = Literal["mcq", "theory", "coding"]
DifficultyLevel = Literal["beginner", "intermediate", "advanced"]


@dataclass
class GeneratorConfig:
    """All parameters that control a single generation run."""

    question_type: QuestionType = "mcq"
    difficulty: DifficultyLevel = "intermediate"
    num_questions: int = 10
    subject: str = ""
    # RAG
    retrieval_k: int = 10
    # LLM
    model: str = "gpt-4o"                    # or "gemini-1.5-pro"
    temperature: float = 0.4
    max_tokens: int = 4096
    # Behaviour
    max_retries: int = 3
    parse_retries: int = 2                   # retries on bad JSON


@dataclass
class QuestionDict:
    """
    Canonical question shape.
    Stored verbatim in PostgreSQL `questions` table.
    """

    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    cognitive_category: Literal["conceptual", "practical"]  # for VARK tagging
    options: list[str] = field(default_factory=list)        # MCQ only, 4 items
    correct_answer: str = ""                                # letter A-D or prose
    explanation: str = ""
    source_chunks: list[str] = field(default_factory=list) # chunk IDs used

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_text": self.question_text,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "cognitive_category": self.cognitive_category,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "source_chunks": self.source_chunks,
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert educational assessment designer.
Your task is to generate high-quality exam questions grounded ONLY in the
provided context passages. Never invent facts not present in the context.

RULES:
1. Every question must be answerable from the context alone.
2. Cognitive category:
   - "conceptual" → recall, definition, explain (tests understanding)
   - "practical"  → apply, calculate, design, trace (tests application)
   Distribute roughly 60 % conceptual / 40 % practical.
3. Difficulty aligns with the requested level: {difficulty}.
4. Return ONLY a raw JSON array — no markdown fences, no preamble.

JSON schema for each element:
{{
  "question_text": "<string>",
  "cognitive_category": "conceptual" | "practical",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],  // MCQ only; [] for theory/coding
  "correct_answer": "<A|B|C|D for MCQ, prose for theory, expected output for coding>",
  "explanation": "<why this answer is correct, 1-3 sentences>"
}}
"""

_USER_PROMPT_TEMPLATE = """\
CONTEXT PASSAGES (retrieved from the uploaded document):
---
{context}
---

TASK:
Subject: {subject}
Question type: {question_type}
Difficulty: {difficulty}
Number of questions to generate: {num_questions}

{type_specific_instructions}

Generate exactly {num_questions} questions now.
"""

_TYPE_INSTRUCTIONS: dict[QuestionType, str] = {
    "mcq": (
        "Each question must have exactly 4 options labelled A, B, C, D. "
        "Only one option is correct. Make distractors plausible but clearly wrong "
        "to an expert. The 'options' array must contain all four strings."
    ),
    "theory": (
        "Each question is open-ended. The 'options' field must be an empty list []. "
        "The 'correct_answer' is a model answer in 2-5 sentences that a marker would "
        "use to grade the student. Avoid yes/no questions."
    ),
    "coding": (
        "Each question presents a programming task. The 'options' field must be []. "
        "The 'correct_answer' field contains the expected output or a reference "
        "solution. Include the function signature or stub in 'question_text'."
    ),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuestionGenerator:
    """
    Orchestrates RAG retrieval → prompt construction → LLM call → parsing.

    Usage
    -----
        store = VectorStore.load(session_id)
        gen   = QuestionGenerator(store)
        qs    = gen.generate(config)
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.store = vector_store
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, config: GeneratorConfig) -> list[QuestionDict]:
        """
        Full pipeline: retrieve → prompt → call → parse → validate.
        Returns a list of QuestionDict objects.
        """
        logger.info(
            "Generating %d %s questions | difficulty=%s | subject=%r",
            config.num_questions, config.question_type,
            config.difficulty, config.subject,
        )

        # Step 1 — retrieve relevant chunks via RAG
        chunks = self._retrieve_chunks(config)
        logger.debug("Retrieved %d chunks from vector store", len(chunks))

        # Step 2 — build the prompt
        prompt = self._build_prompt(chunks, config)

        # Step 3 — call LLM with retry logic
        raw = self._call_llm(prompt, config)

        # Step 4 — parse + validate JSON
        questions = self._parse_and_validate(raw, config, chunks)

        logger.info("Successfully generated %d questions", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Step 1: RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_chunks(self, config: GeneratorConfig) -> list[dict[str, Any]]:
        """
        Query the vector store with the subject string.
        Returns list of {"chunk_id", "text", "metadata"} dicts.
        """
        query = config.subject or "key concepts and topics"
        results = self.store.search(query=query, k=config.retrieval_k)

        if not results:
            logger.warning("Vector store returned 0 chunks — check index for this session")
            raise ValueError("No chunks retrieved from vector store. Was the document ingested?")

        return results

    # ------------------------------------------------------------------
    # Step 2: Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        chunks: list[dict[str, Any]],
        config: GeneratorConfig,
    ) -> tuple[str, str]:
        """
        Returns (system_prompt, user_prompt) tuple.
        Context is assembled from retrieved chunks, numbered for traceability.
        """
        # Assemble context block
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            heading = meta.get("heading", f"Passage {i}")
            context_parts.append(f"[{i}] {heading}\n{chunk['text'].strip()}")
        context_block = "\n\n".join(context_parts)

        system = _SYSTEM_PROMPT.format(difficulty=config.difficulty)

        user = _USER_PROMPT_TEMPLATE.format(
            context=context_block,
            subject=config.subject or "the uploaded document",
            question_type=config.question_type,
            difficulty=config.difficulty,
            num_questions=config.num_questions,
            type_specific_instructions=_TYPE_INSTRUCTIONS[config.question_type],
        )

        return system, user

    # ------------------------------------------------------------------
    # Step 3: LLM call
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_llm(
        self,
        prompt: tuple[str, str],
        config: GeneratorConfig,
    ) -> str:
        """
        Calls OpenAI chat completions.
        Returns raw text content of the assistant message.
        """
        system_prompt, user_prompt = prompt

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},  # GPT-4o JSON mode
        )
        elapsed = time.perf_counter() - start

        raw = response.choices[0].message.content or ""
        logger.debug(
            "LLM call complete | model=%s | tokens=%d | elapsed=%.2fs",
            config.model,
            response.usage.total_tokens if response.usage else -1,
            elapsed,
        )
        return raw

    # ------------------------------------------------------------------
    # Step 4: Parse + validate
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: str,
        config: GeneratorConfig,
        chunks: list[dict[str, Any]],
    ) -> list[QuestionDict]:
        """
        Parses the LLM JSON response and builds QuestionDict objects.
        Handles common JSON quirks (fences, trailing commas, wrapped objects).
        """
        chunk_ids = [c.get("chunk_id", "") for c in chunks]

        data = self._safe_json_parse(raw)

        # LLM sometimes wraps in {"questions": [...]} — unwrap it
        if isinstance(data, dict):
            for key in ("questions", "items", "results", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # Try the first list value we find
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array from LLM, got: {type(data).__name__}")

        questions: list[QuestionDict] = []
        for i, item in enumerate(data):
            try:
                q = self._validate_item(item, config, chunk_ids, index=i)
                questions.append(q)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping question %d due to validation error: %s", i, exc)

        if not questions:
            raise ValueError("LLM returned 0 valid questions after validation.")

        # Trim or warn if count mismatch
        if len(questions) < config.num_questions:
            logger.warning(
                "Requested %d questions but only %d passed validation",
                config.num_questions, len(questions),
            )

        return questions[: config.num_questions]

    def _validate_item(
        self,
        item: dict[str, Any],
        config: GeneratorConfig,
        chunk_ids: list[str],
        index: int,
    ) -> QuestionDict:
        """Validates one question dict from the LLM and returns a QuestionDict."""

        # Required field
        text = str(item.get("question_text", "")).strip()
        if not text:
            raise ValueError(f"item[{index}] has empty question_text")

        # Cognitive category — default to conceptual if missing/invalid
        category = item.get("cognitive_category", "conceptual")
        if category not in ("conceptual", "practical"):
            category = "conceptual"

        # Options — enforce 4 items for MCQ
        options = item.get("options", [])
        if config.question_type == "mcq":
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(
                    f"item[{index}] MCQ must have exactly 4 options, got {len(options)}"
                )
            options = [str(o).strip() for o in options]
        else:
            options = []

        # Correct answer
        answer = str(item.get("correct_answer", "")).strip()
        if not answer:
            raise ValueError(f"item[{index}] missing correct_answer")

        if config.question_type == "mcq":
            # Normalise to single uppercase letter
            letter = answer.strip().upper()[:1]
            if letter not in ("A", "B", "C", "D"):
                raise ValueError(f"item[{index}] MCQ answer '{answer}' is not A-D")
            answer = letter

        explanation = str(item.get("explanation", "")).strip()

        return QuestionDict(
            question_text=text,
            question_type=config.question_type,
            difficulty=config.difficulty,
            cognitive_category=category,
            options=options,
            correct_answer=answer,
            explanation=explanation,
            source_chunks=chunk_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json_parse(raw: str) -> Any:
        """
        Tolerant JSON parser.
        Strips markdown fences, removes trailing commas, then parses.
        """
        # Strip ```json ... ``` fences
        text = re.sub(r"```(?:json)?\s*", "", raw).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas before ] or } (common LLM mistake)
        text_fixed = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(text_fixed)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed after cleanup. Raw snippet:\n%s", text[:500])
            raise ValueError(f"Could not parse LLM response as JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience factory — used by Celery task
# ---------------------------------------------------------------------------

def build_generator(session_id: str) -> QuestionGenerator:
    """
    Loads the vector store for the given session and returns a ready generator.
    Called from ml/tasks/qgen_task.py.
    """
    store = VectorStore.load(session_id)
    return QuestionGenerator(store)