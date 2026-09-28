"""Verify compact OpenClaw usage survives successful and failed turns."""

import json

from agent_eval.openshell.turn_usage import extract_openclaw_turn_usage


def _line(kind, data, ts):
    return json.dumps({"type": kind, "ts": ts, "data": data})


def test_assistant_turns_keep_input_output_reasoning_and_failure_status():
    transcript = "\n".join([
        _line("assistant.message", {"message": {"role": "assistant", "usage": {
            "input": 100, "output": 50, "reasoningTokens": 40,
            "cacheRead": 200, "cacheWrite": 0}, "stopReason": "toolUse"}}, "t1"),
        _line("model.completed", {"stopReason": "error", "promptCache": {
            "lastCallUsage": {"input": 110, "output": 16384, "reasoningTokens": 16000}}}, "t2"),
        _line("assistant.message", {"message": {"role": "assistant", "usage": {
            "input": 110, "output": 16384, "reasoningTokens": 16000},
            "stopReason": "error"}}, "t3"),
    ])
    rows = extract_openclaw_turn_usage(transcript, trajectory=True)["turns"]
    assert len(rows) == 2
    assert rows[0]["input_tokens"] == 100
    assert rows[0]["output_tokens"] == 50
    assert rows[0]["reasoning_tokens"] == 40
    assert rows[0]["cache_read_tokens"] == 200
    assert rows[1]["output_tokens"] == 16384
    assert rows[1]["reasoning_tokens"] == 16000
    assert rows[1]["stop_reason"] == "error"


def test_failed_final_call_without_assistant_message_uses_last_call_usage():
    transcript = "\n".join([
        _line("assistant.message", {"message": {"role": "assistant", "usage": {
            "input": 10, "output": 20}, "stopReason": "toolUse"}}, "t1"),
        _line("model.completed", {"stopReason": "error", "usage": {
            "input": 1000, "output": 2000}, "promptCache": {"lastCallUsage": {
            "input": 30, "output": 40, "reasoningTokens": 35}}}, "t2"),
    ])
    rows = extract_openclaw_turn_usage(transcript, trajectory=True)["turns"]
    assert len(rows) == 2
    assert rows[1]["source"] == "model.completed.lastCallUsage"
    assert rows[1]["input_tokens"] == 30
    assert rows[1]["output_tokens"] == 40
    assert rows[1]["reasoning_tokens"] == 35


def test_missing_failed_call_usage_is_explicitly_unknown():
    transcript = _line("model.completed", {"stopReason": "error", "usage": {
        "input": 1000, "output": 2000}}, "t1")
    row = extract_openclaw_turn_usage(transcript, trajectory=True)["turns"][0]
    assert row["source"] == "model.completed.usage-unavailable"
    assert row["output_tokens"] is None


def test_failed_assistant_without_usage_is_filled_without_duplicate_turn():
    transcript = "\n".join([
        _line("model.completed", {"stopReason": "error", "promptCache": {
            "lastCallUsage": {"input": 30, "output": 40, "reasoningTokens": 35}}}, "t1"),
        _line("assistant.message", {"message": {"role": "assistant",
            "stopReason": "error"}}, "t2"),
    ])
    rows = extract_openclaw_turn_usage(transcript, trajectory=True)["turns"]
    assert len(rows) == 1
    assert rows[0]["output_tokens"] == 40
    assert rows[0]["source"] == "model.completed.lastCallUsage"


def test_legacy_session_message_usage():
    line = json.dumps({"type": "message", "timestamp": "t1", "message": {
        "role": "assistant", "usage": {"input": 3, "output": 5,
        "reasoningTokens": 2}, "stopReason": "stop"}})
    row = extract_openclaw_turn_usage(line, trajectory=False)["turns"][0]
    assert (row["input_tokens"], row["output_tokens"], row["reasoning_tokens"]) == (3, 5, 2)


def test_evaluate_logs_turns_and_case_total_without_changing_artifact():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from agent_eval.openshell.run import _log_openclaw_token_usage

    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "openclaw-turn-usage.json"
        artifact.write_text(json.dumps({"schema_version": 1, "turns": [
            {"turn": 1, "input_tokens": 100, "output_tokens": 20,
             "reasoning_tokens": 15, "cache_read_tokens": 200,
             "stop_reason": "toolUse", "source": "assistant.message"},
            {"turn": 2, "input_tokens": 30, "output_tokens": 16384,
             "reasoning_tokens": 16000, "stop_reason": "error",
             "source": "model.completed.lastCallUsage"},
        ]}))
        before = artifact.read_bytes()
        with patch("agent_eval.openshell.run.logger") as log:
            _log_openclaw_token_usage(Path(directory), "morning-briefing",
                                      {"token_usage": {"input": 130, "output": 16404}})
        records = [json.loads(call.args[1]) for call in log.info.call_args_list]
        assert len(records) == 3
        assert records[0]["scope"] == "turn" and records[0]["cache_read_tokens"] == 200
        assert records[1]["output_tokens"] == 16384
        assert records[1]["reasoning_tokens"] == 16000
        assert records[1]["stop_reason"] == "error"
        assert records[2]["scope"] == "case"
        assert records[2]["output_tokens"] == 16404
        assert artifact.read_bytes() == before


def test_evaluate_logs_unknown_turn_values_without_text():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from agent_eval.openshell.run import _log_openclaw_token_usage

    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "openclaw-turn-usage.json"
        artifact.write_text(json.dumps({"turns": [
            {"turn": 1, "input_tokens": "private prompt", "output_tokens": None,
             "reasoning_tokens": None, "stop_reason": "private response",
             "source": "model.completed.usage-unavailable"},
        ]}))
        with patch("agent_eval.openshell.run.logger") as log:
            _log_openclaw_token_usage(Path(directory), "morning-briefing",
                                      {"token_usage": {"input": 0, "output": 0}})
        records = [json.loads(call.args[1]) for call in log.info.call_args_list]
        assert records[0]["input_tokens"] is None
        assert records[0]["output_tokens"] is None
        assert records[0]["stop_reason"] == "other"
        assert "private" not in str(records)
