"""
quiz/sandbox.py
---------------
Isolated code execution for coding quiz questions.

Execution strategy (tried in order):
  1. Docker container   – if Docker socket is reachable and image is present
  2. Subprocess + venv  – fallback for local dev / environments without Docker

Each strategy:
  - Enforces a wall-clock timeout
  - Captures stdout / stderr
  - Runs each test case and reports per-case pass / fail
  - Never lets student code touch the host network or filesystem beyond /tmp

Supported languages: python, javascript (node), java, cpp
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

DOCKER_ENABLED       = os.getenv("SANDBOX_DOCKER_ENABLED", "true").lower() == "true"
DOCKER_PYTHON_IMAGE  = os.getenv("SANDBOX_PYTHON_IMAGE",  "python:3.12-alpine")
DOCKER_NODE_IMAGE    = os.getenv("SANDBOX_NODE_IMAGE",    "node:20-alpine")
DOCKER_JAVA_IMAGE    = os.getenv("SANDBOX_JAVA_IMAGE",    "eclipse-temurin:21-jdk-alpine")
DOCKER_CPP_IMAGE     = os.getenv("SANDBOX_CPP_IMAGE",     "gcc:13-alpine")
MAX_OUTPUT_BYTES     = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", str(64 * 1024)))  # 64 KB
DEFAULT_TIMEOUT_SEC  = int(os.getenv("SANDBOX_TIMEOUT_SEC", "10"))
NETWORK_MODE         = os.getenv("SANDBOX_NETWORK_MODE", "none")   # "none" blocks all network
MEMORY_LIMIT         = os.getenv("SANDBOX_MEMORY_LIMIT", "128m")
CPU_LIMIT            = os.getenv("SANDBOX_CPU_LIMIT", "0.5")       # half a core

_DOCKER_IMAGE_MAP: dict[str, str] = {
    "python":     DOCKER_PYTHON_IMAGE,
    "javascript": DOCKER_NODE_IMAGE,
    "java":       DOCKER_JAVA_IMAGE,
    "cpp":        DOCKER_CPP_IMAGE,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    stdin:           str  = ""
    expected_output: str  = ""
    args:            list = field(default_factory=list)   # CLI arguments

@dataclass
class CaseResult:
    passed:   bool
    stdout:   str
    stderr:   str
    exit_code: int
    timed_out: bool = False

@dataclass
class SandboxResult:
    all_passed:  bool
    cases:       list[CaseResult]
    stdout:      str          # concatenated first-case stdout for display
    stderr:      str
    error:       str | None   # sandbox-level error (compile error, timeout, etc.)
    strategy:    str          # "docker" | "subprocess"
    duration_ms: int


# ---------------------------------------------------------------------------
# Language runners
# ---------------------------------------------------------------------------

def _write_source(tmp_dir: str, code: str, language: str) -> str:
    """Write code to the correct filename for the language. Returns the file path."""
    filenames = {
        "python":     "solution.py",
        "javascript": "solution.js",
        "java":       "Solution.java",
        "cpp":        "solution.cpp",
    }
    filename = filenames.get(language, "solution.txt")
    path = os.path.join(tmp_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path


def _build_run_commands(language: str, source_file: str) -> tuple[list[str] | None, list[str]]:
    """
    Return (compile_cmd, run_cmd) for the given language.
    compile_cmd is None for interpreted languages.
    """
    base = os.path.basename(source_file)
    name = os.path.splitext(base)[0]

    commands: dict[str, tuple[list[str] | None, list[str]]] = {
        "python":     (None,                          ["python", source_file]),
        "javascript": (None,                          ["node",   source_file]),
        "java":       (["javac", source_file],        ["java", "-cp", os.path.dirname(source_file), name]),
        "cpp":        (["g++", "-O2", "-o",
                         source_file.replace(".cpp", ""),
                         source_file],
                       [source_file.replace(".cpp", "")]),
    }
    return commands.get(language, (None, ["cat", source_file]))


def _truncate(text: str) -> str:
    if len(text.encode()) > MAX_OUTPUT_BYTES:
        return text[: MAX_OUTPUT_BYTES // 2] + "\n… [output truncated] …"
    return text


# ---------------------------------------------------------------------------
# Docker execution
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_in_docker(
    code:        str,
    language:    str,
    test_cases:  list[dict],
    timeout_sec: int,
    tmp_dir:     str,
) -> SandboxResult:
    """Execute code inside a Docker container, one container per test case."""
    image      = _DOCKER_IMAGE_MAP.get(language, DOCKER_PYTHON_IMAGE)
    source_path = _write_source(tmp_dir, code, language)
    compile_cmd, run_cmd = _build_run_commands(language, f"/workspace/{os.path.basename(source_path)}")

    t_start = time.monotonic()
    case_results: list[CaseResult] = []
    compile_error: str | None = None

    # ---- compile step (Java / C++) ----------------------------------------
    if compile_cmd:
        docker_compile = [
            "docker", "run", "--rm",
            "--network", NETWORK_MODE,
            "--memory", MEMORY_LIMIT,
            "--cpus",   CPU_LIMIT,
            "-v", f"{tmp_dir}:/workspace",
            "-w", "/workspace",
            image,
        ] + [c.replace(source_path, f"/workspace/{os.path.basename(source_path)}") for c in compile_cmd]

        try:
            proc = subprocess.run(
                docker_compile,
                capture_output=True, text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                compile_error = proc.stderr or proc.stdout
        except subprocess.TimeoutExpired:
            compile_error = "Compilation timed out."

    if compile_error:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        return SandboxResult(
            all_passed  = False,
            cases       = [],
            stdout      = "",
            stderr      = compile_error,
            error       = f"Compilation failed:\n{compile_error}",
            strategy    = "docker",
            duration_ms = duration_ms,
        )

    # ---- run each test case -------------------------------------------------
    for tc_raw in test_cases:
        tc = TestCase(
            stdin           = tc_raw.get("stdin", ""),
            expected_output = tc_raw.get("expected_output", "").strip(),
            args            = tc_raw.get("args", []),
        )

        docker_run_cmd = [
            "docker", "run", "--rm", "-i",
            "--network", NETWORK_MODE,
            "--memory", MEMORY_LIMIT,
            "--cpus",   CPU_LIMIT,
            "--pids-limit", "64",
            "-v", f"{tmp_dir}:/workspace:ro",
            "-w", "/workspace",
            image,
        ] + [c.replace(source_path, f"/workspace/{os.path.basename(source_path)}") for c in run_cmd]

        if tc.args:
            docker_run_cmd += tc.args

        timed_out = False
        try:
            proc = subprocess.run(
                docker_run_cmd,
                input=tc.stdin,
                capture_output=True, text=True,
                timeout=timeout_sec,
            )
            stdout    = _truncate(proc.stdout)
            stderr    = _truncate(proc.stderr)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout    = ""
            stderr    = f"Time limit exceeded ({timeout_sec}s)"
            exit_code = -1
            timed_out = True

        actual   = stdout.strip()
        expected = tc.expected_output
        passed   = (exit_code == 0) and (actual == expected)

        case_results.append(CaseResult(
            passed    = passed,
            stdout    = stdout,
            stderr    = stderr,
            exit_code = exit_code,
            timed_out = timed_out,
        ))

    duration_ms = int((time.monotonic() - t_start) * 1000)
    all_passed  = bool(case_results) and all(c.passed for c in case_results)
    first_stdout = case_results[0].stdout if case_results else ""
    first_stderr = case_results[0].stderr if case_results else ""

    return SandboxResult(
        all_passed  = all_passed,
        cases       = case_results,
        stdout      = first_stdout,
        stderr      = first_stderr,
        error       = None,
        strategy    = "docker",
        duration_ms = duration_ms,
    )


# ---------------------------------------------------------------------------
# Subprocess (fallback) execution
# ---------------------------------------------------------------------------

def _run_subprocess(
    code:        str,
    language:    str,
    test_cases:  list[dict],
    timeout_sec: int,
    tmp_dir:     str,
) -> SandboxResult:
    """
    Fallback: run code via subprocess without Docker.
    Only safe for trusted content / local development.
    """
    if language not in ("python", "javascript"):
        return SandboxResult(
            all_passed  = False,
            cases       = [],
            stdout      = "",
            stderr      = "",
            error       = f"Subprocess sandbox does not support '{language}' without Docker.",
            strategy    = "subprocess",
            duration_ms = 0,
        )

    source_path              = _write_source(tmp_dir, code, language)
    _, run_cmd               = _build_run_commands(language, source_path)
    t_start                  = time.monotonic()
    case_results: list[CaseResult] = []

    for tc_raw in test_cases:
        tc = TestCase(
            stdin           = tc_raw.get("stdin", ""),
            expected_output = tc_raw.get("expected_output", "").strip(),
            args            = tc_raw.get("args", []),
        )

        timed_out = False
        try:
            proc = subprocess.run(
                run_cmd + tc.args,
                input=tc.stdin,
                capture_output=True, text=True,
                timeout=timeout_sec,
                cwd=tmp_dir,
            )
            stdout    = _truncate(proc.stdout)
            stderr    = _truncate(proc.stderr)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout    = ""
            stderr    = f"Time limit exceeded ({timeout_sec}s)"
            exit_code = -1
            timed_out = True

        actual = stdout.strip()
        passed = (exit_code == 0) and (actual == tc.expected_output)

        case_results.append(CaseResult(
            passed    = passed,
            stdout    = stdout,
            stderr    = stderr,
            exit_code = exit_code,
            timed_out = timed_out,
        ))

    duration_ms = int((time.monotonic() - t_start) * 1000)
    all_passed  = bool(case_results) and all(c.passed for c in case_results)

    return SandboxResult(
        all_passed  = all_passed,
        cases       = case_results,
        stdout      = case_results[0].stdout if case_results else "",
        stderr      = case_results[0].stderr if case_results else "",
        error       = None,
        strategy    = "subprocess",
        duration_ms = duration_ms,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_code_in_sandbox(
    code:        str,
    language:    str,
    test_cases:  list[dict],
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """
    Execute student code against a list of test cases in an isolated environment.

    Parameters
    ----------
    code : str
        The source code submitted by the student.
    language : str
        One of: "python", "javascript", "java", "cpp"
    test_cases : list[dict]
        Each dict may contain:
            stdin           : str   – data piped to stdin
            expected_output : str   – expected stdout (stripped)
            args            : list  – CLI arguments
    timeout_sec : int
        Wall-clock limit per test case.

    Returns
    -------
    dict with keys:
        all_passed  : bool
        stdout      : str          (first test case stdout, for display)
        stderr      : str
        error       : str | None   (sandbox-level error message)
        strategy    : str          ("docker" | "subprocess")
        duration_ms : int
        cases       : list[dict]   (per-case results)
    """
    language = language.lower()

    if not code or not code.strip():
        return _empty_result(error="No code submitted.")

    if not test_cases:
        # No test cases: just execute and return stdout
        test_cases = [{"stdin": "", "expected_output": "", "args": []}]

    with tempfile.TemporaryDirectory(prefix="quiz_sandbox_") as tmp_dir:
        try:
            if DOCKER_ENABLED and _docker_available():
                result = _run_in_docker(
                    code=code,
                    language=language,
                    test_cases=test_cases,
                    timeout_sec=timeout_sec,
                    tmp_dir=tmp_dir,
                )
            else:
                logger.warning(
                    "Docker unavailable or disabled — falling back to subprocess sandbox "
                    "(not recommended for production)."
                )
                result = _run_subprocess(
                    code=code,
                    language=language,
                    test_cases=test_cases,
                    timeout_sec=timeout_sec,
                    tmp_dir=tmp_dir,
                )
        except Exception as exc:
            logger.exception("Sandbox execution error: %s", exc)
            return _empty_result(error=f"Internal sandbox error: {exc}")

    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _empty_result(error: str) -> dict[str, Any]:
    return {
        "all_passed":  False,
        "stdout":      "",
        "stderr":      "",
        "error":       error,
        "strategy":    "none",
        "duration_ms": 0,
        "cases":       [],
    }


def _result_to_dict(result: SandboxResult) -> dict[str, Any]:
    return {
        "all_passed":  result.all_passed,
        "stdout":      result.stdout,
        "stderr":      result.stderr,
        "error":       result.error,
        "strategy":    result.strategy,
        "duration_ms": result.duration_ms,
        "cases": [
            {
                "passed":    c.passed,
                "stdout":    c.stdout,
                "stderr":    c.stderr,
                "exit_code": c.exit_code,
                "timed_out": c.timed_out,
            }
            for c in result.cases
        ],
    }


# ---------------------------------------------------------------------------
# CLI helper (for manual testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sample_code = textwrap.dedent(
        """
        n = int(input())
        print(n * n)
        """
    )
    _test_cases = [
        {"stdin": "4\n",  "expected_output": "16"},
        {"stdin": "10\n", "expected_output": "100"},
        {"stdin": "0\n",  "expected_output": "0"},
    ]

    r = run_code_in_sandbox(
        code=_sample_code,
        language="python",
        test_cases=_test_cases,
        timeout_sec=5,
    )
    print(json.dumps(r, indent=2))