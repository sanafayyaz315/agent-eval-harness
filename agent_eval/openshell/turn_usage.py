"""Extract numeric per-turn usage from OpenClaw session and trajectory exports."""

from __future__ import annotations

import json


def _usage_row(usage: object, *, turn: int, timestamp: str | None,
               stop_reason: str | None, source: str) -> dict:
    usage = usage if isinstance(usage, dict) else {}
    return {
        "turn": turn,
        "timestamp": timestamp,
        "input_tokens": usage.get("input"),
        "output_tokens": usage.get("output"),
        "reasoning_tokens": usage.get("reasoningTokens"),
        "cache_read_tokens": usage.get("cacheRead"),
        "cache_write_tokens": usage.get("cacheWrite"),
        "stop_reason": stop_reason,
        "source": source,
    }


def extract_openclaw_turn_usage(jsonl_text: str, *, trajectory: bool) -> dict:
    """Return only usage metadata, including a failed final call when observed.

    ``lastCallUsage`` is used only when the final model call has no matching
    assistant message. Missing counts remain null; run totals are never
    presented as counts for one turn.
    """
    turns: list[dict] = []
    final_call: dict | None = None
    for line in jsonl_text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        message = data.get("message") if trajectory else event.get("message")
        if kind == ("assistant.message" if trajectory else "message") and isinstance(message, dict) and message.get("role") == "assistant":
            turns.append(_usage_row(
                message.get("usage"), turn=len(turns) + 1,
                timestamp=event.get("ts") or event.get("timestamp"),
                stop_reason=message.get("stopReason"),
                source="assistant.message" if trajectory else "session.message",
            ))
        elif trajectory and kind == "model.completed":
            cache = data.get("promptCache") if isinstance(data.get("promptCache"), dict) else {}
            final_call = {
                "timestamp": event.get("ts") or event.get("timestamp"),
                "usage": cache.get("lastCallUsage"),
                "stop_reason": data.get("stopReason"),
            }

    if final_call and final_call["stop_reason"] in ("error", "length", "aborted"):
        usage = final_call["usage"]
        last = turns[-1] if turns else None
        same_last_call = (
            last is not None and isinstance(usage, dict)
            and last["input_tokens"] == usage.get("input")
            and last["output_tokens"] == usage.get("output")
            and last["reasoning_tokens"] == usage.get("reasoningTokens")
        )
        missing_last_usage = (
            last is not None and last["stop_reason"] == final_call["stop_reason"]
            and last["input_tokens"] is None and last["output_tokens"] is None
        )
        if missing_last_usage:
            turns[-1] = _usage_row(
                usage, turn=last["turn"], timestamp=last["timestamp"],
                stop_reason=last["stop_reason"],
                source="model.completed.lastCallUsage" if isinstance(usage, dict) else "model.completed.usage-unavailable",
            )
        elif not same_last_call:
            turns.append(_usage_row(
                usage, turn=len(turns) + 1,
                timestamp=final_call["timestamp"],
                stop_reason=final_call["stop_reason"],
                source="model.completed.lastCallUsage" if isinstance(usage, dict) else "model.completed.usage-unavailable",
            ))

    return {"schema_version": 1, "turns": turns}
