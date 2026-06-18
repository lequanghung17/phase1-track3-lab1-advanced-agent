from __future__ import annotations

import json
import hashlib
import os
import time

from dotenv import load_dotenv
from google import genai

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


load_dotenv()
_GEMINI_CLIENT: genai.Client | None = None
_LAST_LLM_CALL_AT = 0.0
_RUNTIME_TOKENS = 0
_RUNTIME_LATENCY_MS = 0


def reset_runtime_metrics() -> None:
    global _RUNTIME_TOKENS, _RUNTIME_LATENCY_MS
    _RUNTIME_TOKENS = 0
    _RUNTIME_LATENCY_MS = 0


def get_runtime_metrics() -> tuple[int, int]:
    return _RUNTIME_TOKENS, _RUNTIME_LATENCY_MS


def _record_runtime_metrics(tokens: int, latency_ms: int) -> None:
    global _RUNTIME_TOKENS, _RUNTIME_LATENCY_MS
    _RUNTIME_TOKENS += max(0, tokens)
    _RUNTIME_LATENCY_MS += max(0, latency_ms)


def _estimate_tokens(*texts: str) -> int:
    text = " ".join(texts)
    return max(1, round(len(text.split()) * 1.3))


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def _mock_should_fail_first_attempt(example: QAExample) -> bool:
    digest = hashlib.md5(example.qid.encode("utf-8")).hexdigest()
    return int(digest, 16) % 3 == 0


def _mock_wrong_answer(example: QAExample) -> str:
    if example.context:
        return example.context[0].title
    return "Unknown"


def _mock_actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    started = time.monotonic()
    if not _mock_should_fail_first_attempt(example):
        answer = example.gold_answer
    elif agent_type == "react":
        answer = _mock_wrong_answer(example)
    elif attempt_id == 1 and not reflection_memory:
        answer = _mock_wrong_answer(example)
    else:
        answer = example.gold_answer

    context_text = format_context(example)
    _record_runtime_metrics(
        _estimate_tokens(example.question, context_text, " ".join(reflection_memory), answer),
        round((time.monotonic() - started) * 1000),
    )
    return answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    started = time.monotonic()
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        result = JudgeResult(
            score=1,
            reason="Final answer matches the gold answer after normalization.",
        )
    else:
        result = JudgeResult(
            score=0,
            reason="The predicted answer does not match the gold answer.",
            missing_evidence=["Re-check the supporting context and complete all reasoning hops."],
            spurious_claims=[answer],
        )
    _record_runtime_metrics(
        _estimate_tokens(example.question, example.gold_answer, answer, result.reason),
        round((time.monotonic() - started) * 1000),
    )
    return result


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    started = time.monotonic()
    result = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="Do not stop at a distractor entity; verify the final answer against the supporting facts.",
        next_strategy="Identify the relevant supporting paragraphs first, then answer with the final entity only.",
    )
    _record_runtime_metrics(
        _estimate_tokens(example.question, judge.reason, result.lesson, result.next_strategy),
        round((time.monotonic() - started) * 1000),
    )
    return result


def _client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it in PowerShell with "
            '$env:GEMINI_API_KEY="your-key" or put it in a .env file.'
        )
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def call_llm(system_prompt: str, user_prompt: str) -> str:
    global _LAST_LLM_CALL_AT

    delay_seconds = float(os.getenv("LLM_CALL_DELAY_SECONDS", "7"))
    elapsed = time.monotonic() - _LAST_LLM_CALL_AT
    if elapsed < delay_seconds:
        time.sleep(delay_seconds - elapsed)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    started = time.monotonic()
    response = _client().models.generate_content(
        model=model,
        contents=f"{system_prompt.strip()}\n\n{user_prompt.strip()}",
    )
    latency_ms = round((time.monotonic() - started) * 1000)
    _LAST_LLM_CALL_AT = time.monotonic()
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    usage = getattr(response, "usage_metadata", None)
    total_tokens = getattr(usage, "total_token_count", None) if usage else None
    if total_tokens is None:
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = prompt_tokens + output_tokens
    if not total_tokens:
        total_tokens = _estimate_tokens(system_prompt, user_prompt, response.text)

    _record_runtime_metrics(int(total_tokens), latency_ms)
    return response.text.strip()


def format_context(example: QAExample) -> str:
    return "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}"
        for chunk in example.context
    )


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return a JSON object: {text}")
    return json.loads(cleaned[start : end + 1])


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if _provider() == "mock":
        return _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)

    memory_text = "\n".join(f"- {item}" for item in reflection_memory) or "None"
    user_prompt = f"""
Question:
{example.question}

Context:
{format_context(example)}

Attempt:
{attempt_id}

Agent type:
{agent_type}

Reflection memory:
{memory_text}
"""
    return call_llm(ACTOR_SYSTEM, user_prompt)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if _provider() == "mock":
        return _mock_evaluator(example, answer)

    user_prompt = f"""
Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}

Context:
{format_context(example)}

Evaluate whether the predicted answer is correct.
Return only valid JSON.
"""
    data = parse_json_object(call_llm(EVALUATOR_SYSTEM, user_prompt))
    return JudgeResult(**data)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if _provider() == "mock":
        return _mock_reflector(example, attempt_id, judge)

    user_prompt = f"""
Question:
{example.question}

Context:
{format_context(example)}

Failed attempt id:
{attempt_id}

Evaluator feedback:
{judge.reason}

Missing evidence:
{judge.missing_evidence}

Spurious claims:
{judge.spurious_claims}

Reflect on the mistake and propose a better next strategy.
Return only valid JSON.
"""
    data = parse_json_object(call_llm(REFLECTOR_SYSTEM, user_prompt))
    data["attempt_id"] = attempt_id
    return ReflectionEntry(**data)
