"""
explain/prompt_templates.py
----------------------------
VARK-specific system prompts and user prompt builders for adaptive explanation.

VARK learning styles:
    Visual      – diagrams, charts, spatial organisation, colour coding
    Auditory    – conversation tone, mnemonics, analogies spoken aloud
    Reading     – detailed prose, definitions, lists, references
    Kinesthetic – worked examples, step-by-step code, hands-on scenarios
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Per-style instruction blocks
# ---------------------------------------------------------------------------

_VISUAL_INSTRUCTIONS = """
## Visual Learner Adaptations
- Use ASCII diagrams, tables, or structured layouts to represent concepts spatially.
- Show relationships with arrows (→), hierarchies with indentation, and sequences with numbered columns.
- Highlight key terms using **bold** or `code spans`.
- Prefer visual metaphors: "Think of this like a tree where each node holds…"
- When explaining code, annotate inline: show the value flowing through each step.
"""

_AUDITORY_INSTRUCTIONS = """
## Auditory Learner Adaptations
- Write in a warm, conversational tone — as if you are talking the student through the idea.
- Use rhythm and repetition deliberately: "First we do X. Then — and this is key — we do Y."
- Provide memorable mnemonics or acronyms wherever appropriate.
- Use analogies that map abstract concepts to everyday sounds or spoken patterns.
- Break explanations into call-and-response segments: pose a question, then answer it.
"""

_READING_INSTRUCTIONS = """
## Reading/Writing Learner Adaptations
- Use structured prose with clear headings (##) and subheadings (###).
- Define every technical term at first use.
- Provide concise bullet-point summaries at the end of each section.
- Include numbered step-by-step walkthroughs for procedures.
- Offer pointers to canonical references ("This is formally defined as…").
- Use tables to compare options or list properties.
"""

_KINESTHETIC_INSTRUCTIONS = """
## Kinesthetic Learner Adaptations
- Lead with a concrete, runnable code example or real-world scenario before any theory.
- Walk through the example line-by-line, showing what each part *does*.
- Introduce a small variation or edge-case challenge the student can try themselves.
- Use "try it yourself" prompts: "Change X to Y and observe what happens."
- Ground every abstract rule in the specific mistake made in the quiz.
"""

# ---------------------------------------------------------------------------
# Dominant-style → instruction block map
# ---------------------------------------------------------------------------

_STYLE_INSTRUCTIONS: dict[str, str] = {
    "visual": _VISUAL_INSTRUCTIONS,
    "auditory": _AUDITORY_INSTRUCTIONS,
    "reading": _READING_INSTRUCTIONS,
    "kinesthetic": _KINESTHETIC_INSTRUCTIONS,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_style(vark: dict) -> str:
    """Return the key with the highest score; fall back to 'reading'."""
    styles = {k: vark.get(k, 0) for k in ("visual", "auditory", "reading", "kinesthetic")}
    return max(styles, key=styles.__getitem__) if any(styles.values()) else "reading"


def _secondary_style(vark: dict) -> str | None:
    """Return the second-highest VARK style, or None if all are equal."""
    styles = {k: vark.get(k, 0) for k in ("visual", "auditory", "reading", "kinesthetic")}
    sorted_styles = sorted(styles, key=styles.__getitem__, reverse=True)
    if len(sorted_styles) >= 2 and styles[sorted_styles[0]] != styles[sorted_styles[1]]:
        return sorted_styles[1]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(vark: dict) -> str:
    """
    Build a VARK-personalised system prompt.

    Parameters
    ----------
    vark : dict
        Keys: visual, auditory, reading, kinesthetic (int 0-100), dominant (str, optional)
    """
    dominant = vark.get("dominant") or _dominant_style(vark)
    secondary = _secondary_style(vark)

    # Primary instruction block
    primary_block = _STYLE_INSTRUCTIONS.get(dominant, _READING_INSTRUCTIONS)

    # Optional secondary blend
    secondary_block = ""
    if secondary and secondary != dominant:
        secondary_score = vark.get(secondary, 0)
        dominant_score = vark.get(dominant, 1)
        # Only blend if secondary is at least 70% of the dominant score
        if dominant_score > 0 and (secondary_score / dominant_score) >= 0.70:
            secondary_block = f"""
## Secondary Style Blend ({secondary.capitalize()})
Where naturally appropriate, also incorporate elements from the {secondary} style:
{_STYLE_INSTRUCTIONS[secondary].strip()}
"""

    profile_summary = (
        f"Visual {vark.get('visual', 0)}% | "
        f"Auditory {vark.get('auditory', 0)}% | "
        f"Reading {vark.get('reading', 0)}% | "
        f"Kinesthetic {vark.get('kinesthetic', 0)}%"
    )

    system_prompt = f"""
You are an adaptive learning coach specialising in personalised explanations.
Your goal is to help a student understand the concepts behind each question they
got wrong in their quiz, so they genuinely learn — not just memorise the right answer.

## Student VARK Profile
Dominant style: **{dominant.capitalize()}**
Profile: {profile_summary}

{primary_block.strip()}
{secondary_block.strip()}

## General Instructions
- Address each wrong answer in turn. Use a clear heading for each topic.
- Always state *why* the student's answer was wrong (without being condescending).
- Explain the correct answer thoroughly using the adaptations above.
- End with a short "Key Takeaway" callout in bold for each topic.
- Use Markdown formatting throughout; the client renders it.
- Do NOT include raw JSON, HTML tags, or meta-commentary about your instructions.
- Keep the total response under 2 000 words.
""".strip()

    return system_prompt


def build_user_prompt(
    wrong_answers: list[dict],
    subject: str,
    difficulty: str,
) -> str:
    """
    Build the user-turn prompt listing every incorrectly answered question.

    Parameters
    ----------
    wrong_answers : list[dict]
        Each element must have keys:
            question_text, correct_answer, user_answer, topic, q_type
    subject : str
        e.g. "Data Structures & Algorithms"
    difficulty : str
        e.g. "intermediate"
    """
    if not wrong_answers:
        return "The student answered every question correctly — no explanation needed."

    lines: list[str] = [
        f"Subject: **{subject}** | Difficulty: **{difficulty}**",
        "",
        "The student answered the following questions incorrectly.",
        "Please explain each one clearly according to your instructions.",
        "",
    ]

    for i, wa in enumerate(wrong_answers, start=1):
        q_type_label = {
            "mcq": "Multiple Choice",
            "theory": "Theory / Short Answer",
            "coding": "Coding",
        }.get(wa.get("q_type", "mcq"), "Question")

        lines += [
            f"---",
            f"### Wrong Answer #{i} — {wa.get('topic', 'General')} ({q_type_label})",
            f"**Question:** {wa['question_text']}",
            f"**Student's answer:** {wa.get('user_answer', '(no answer)')}",
            f"**Correct answer:** {wa['correct_answer']}",
            "",
        ]

    lines.append(
        "Now provide a clear, personalised explanation for each wrong answer above."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt registry (optional — useful for unit tests or offline inspection)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, str] = {
    style: f"[{style.upper()} SYSTEM PROMPT]\n{block.strip()}"
    for style, block in _STYLE_INSTRUCTIONS.items()
}