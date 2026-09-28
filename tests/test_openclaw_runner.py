"""Tests for the OpenClaw CLI runner."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_eval.agent import RUNNERS
from agent_eval.agent.openclaw import (
    OPENCLAW_EFFORTS,
    OpenClawRunner,
    build_openclaw_argv,
    parse_openclaw_to_case_dict,
    parse_openclaw_to_run_result,
)
from agent_eval.config import EvalConfig


def test_openclaw_is_registered():
    assert RUNNERS["openclaw"] is OpenClawRunner


def test_openclaw_name_property():
    runner = OpenClawRunner()
    assert runner.name == "openclaw"


class TestBuildOpenclawArgv:
    """Tests for build_openclaw_argv helper."""

    def test_minimal_command(self):
        cmd = build_openclaw_argv(model="anthropic/claude-sonnet-4-6")
        assert cmd[:4] == ["openclaw", "agent", "exec", "--json"]
        assert "--model" in cmd
        assert "anthropic/claude-sonnet-4-6" in cmd
        assert "--auth-env-only" in cmd
        assert "--config" not in cmd

    def test_with_cwd(self):
        cmd = build_openclaw_argv(model="m", cwd=Path("/workspace"))
        assert "--cwd" in cmd
        idx = cmd.index("--cwd")
        assert cmd[idx + 1] == "/workspace"

    def test_with_timeout(self):
        cmd = build_openclaw_argv(model="m", timeout_s=300)
        assert "--timeout" in cmd
        idx = cmd.index("--timeout")
        assert cmd[idx + 1] == "300"

    def test_with_effort(self):
        cmd = build_openclaw_argv(model="m", effort="high")
        assert "--thinking" in cmd
        idx = cmd.index("--thinking")
        assert cmd[idx + 1] == "high"

    def test_with_state_dir(self):
        cmd = build_openclaw_argv(model="m", state_dir=Path("/sandbox/.state"))
        assert "--state-dir" in cmd
        idx = cmd.index("--state-dir")
        assert cmd[idx + 1] == "/sandbox/.state"

    def test_without_auth_env_only(self):
        cmd = build_openclaw_argv(model="m", auth_env_only=False)
        assert "--auth-env-only" not in cmd

    def test_with_config_path(self):
        cmd = build_openclaw_argv(
            model="inference/claude-sonnet-4",
            auth_env_only=False,
            config_path=Path("/sandbox/openclaw-eval.json"),
        )
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == "/sandbox/openclaw-eval.json"
        assert "--auth-env-only" not in cmd

    def test_config_path_rejects_auth_env_only(self):
        with pytest.raises(ValueError, match="auth-env-only"):
            build_openclaw_argv(
                model="m",
                auth_env_only=True,
                config_path=Path("/sandbox/openclaw-eval.json"),
            )

    def test_all_options(self):
        cmd = build_openclaw_argv(
            model="anthropic/claude-opus-4-6",
            cwd=Path("/work"),
            timeout_s=600,
            effort="medium",
            state_dir=Path("/state"),
            auth_env_only=False,
            config_path=Path("/sandbox/openclaw-eval.json"),
        )
        assert "--cwd" in cmd
        assert "--timeout" in cmd
        assert "--thinking" in cmd
        assert "--state-dir" in cmd
        assert "--config" in cmd
        assert "--auth-env-only" not in cmd


class TestParseOpenclawEnvelope:
    """Tests for OpenClaw JSON parsing."""

    def test_parse_success_envelope(self):
        stdout = json.dumps({
            "ok": True,
            "costUsd": 0.05,
            "usage": {"input": 100, "output": 50},
            "assistantTurns": 3,
            "model": "claude-sonnet-4-6",
        }).encode()
        result = parse_openclaw_to_run_result(stdout, b"", 0, 1.5)
        
        assert result.exit_code == 0
        assert result.cost_usd == 0.05
        assert result.token_usage == {"input": 100, "output": 50}
        assert result.num_turns == 3
        assert result.resolved_model == "claude-sonnet-4-6"
        assert result.duration_s == 1.5

    def test_parse_error_envelope(self):
        stdout = json.dumps({
            "ok": False,
            "error": {"message": "Rate limited"},
        }).encode()
        result = parse_openclaw_to_run_result(stdout, b"", 1, 2.0)
        
        assert result.exit_code == 1
        assert "Rate limited" in result.stderr

    def test_parse_invalid_json(self):
        result = parse_openclaw_to_run_result(b"not json", b"error output", 1, 0.5)
        
        assert result.exit_code == 1
        assert "error output" in result.stderr
        assert result.token_usage is None

    def test_parse_empty_usage(self):
        stdout = json.dumps({"ok": True}).encode()
        result = parse_openclaw_to_run_result(stdout, b"", 0, 1.0)
        
        assert result.token_usage == {"input": 0, "output": 0}

    def test_stdout_preserved(self):
        stdout = json.dumps({"ok": True, "message": "done"}).encode()
        result = parse_openclaw_to_run_result(stdout, b"", 0, 1.0)
        
        assert result.stdout == stdout.decode()
        assert result.raw_output == {"ok": True, "message": "done"}


class TestParseOpenclawToCaseDict:
    """Tests for case dict parsing (openshell backend)."""

    def test_error_payload_is_not_an_agent_response(self):
        from agent_eval.agent.openclaw import extract_openclaw_response
        stdout = json.dumps({
            "ok": False, "status": "error", "final": "",
            "payloads": [{"text": "LLM request failed: network connection error.", "isError": True}],
        }).encode()
        result = parse_openclaw_to_case_dict(stdout, b"connection error", 1, 3.0)
        assert result["response_text"] == ""
        assert result["exit_code"] == 1
        assert extract_openclaw_response(stdout) == ""

    def test_timeout_final_is_not_an_agent_response(self):
        from agent_eval.agent.openclaw import extract_openclaw_response
        stdout = json.dumps({
            "ok": False, "status": "timeout",
            "final": "Request timed out before a response was generated.",
        }).encode()
        result = parse_openclaw_to_case_dict(stdout, b"", 2, 900.0)
        assert result["response_text"] == ""
        assert result["exit_code"] == 2
        assert extract_openclaw_response(stdout) == ""

    def test_parse_success_to_dict(self):
        stdout = json.dumps({
            "costUsd": 0.10,
            "usage": {"input": 200, "output": 100},
            "assistantTurns": 5,
            "model": "claude-opus-4-6",
        }).encode()
        result = parse_openclaw_to_case_dict(stdout, b"", 0, 3.0)
        
        assert result["exit_code"] == 0
        assert result["duration_s"] == 3.0
        assert result["cost_usd"] == 0.10
        assert result["token_usage"] == {"input": 200, "output": 100}
        assert result["num_turns"] == 5
        assert result["resolved_model"] == "claude-opus-4-6"

    def test_parse_gateway_result_envelope(self):
        stdout = json.dumps({"runId": "r1", "result": {
            "payloads": [{"text": "Brief published"}],
            "meta": {"agentMeta": {"usage": {"input": 12, "output": 3}}},
        }}).encode()
        result = parse_openclaw_to_case_dict(stdout, b"", 0, 3.0)
        assert result["response_text"] == "Brief published"
        assert result["token_usage"] == {"input": 12, "output": 3}

    def test_parse_failure_to_dict(self):
        result = parse_openclaw_to_case_dict(b"bad", b"stderr", 1, 0.5)
        
        assert result["exit_code"] == 1
        assert result["token_usage"] == {"input": 0, "output": 0}
        assert result["stderr"] == "stderr"


class TestOpenClawRunner:
    """Tests for OpenClawRunner class."""

    def test_from_config(self, tmp_path):
        config_path = tmp_path / "eval.yaml"
        config_path.write_text(yaml.safe_dump({
            "name": "openclaw-test",
            "execution": {"prompt": "test prompt"},
            "runner": {
                "type": "openclaw",
                "effort": "medium",
                "env": {"CUSTOM_VAR": "value"},
            },
        }))
        runner = OpenClawRunner.from_config(EvalConfig.from_yaml(config_path))
        
        assert runner._effort == "medium"
        assert runner._env.get("CUSTOM_VAR") == "value"

    def test_from_config_with_overrides(self, tmp_path):
        config_path = tmp_path / "eval.yaml"
        config_path.write_text(yaml.safe_dump({
            "name": "openclaw-test",
            "execution": {"prompt": "test"},
            "runner": {"type": "openclaw", "effort": "low"},
        }))
        runner = OpenClawRunner.from_config(
            EvalConfig.from_yaml(config_path),
            effort="high",
        )
        assert runner._effort == "high"

    def test_invalid_effort_rejected(self, tmp_path):
        config_path = tmp_path / "eval.yaml"
        config_path.write_text(yaml.safe_dump({
            "name": "openclaw-test",
            "execution": {"prompt": "test"},
            "runner": {"type": "openclaw", "effort": "invalid"},
        }))
        with pytest.raises(ValueError, match="Invalid OpenClaw effort"):
            OpenClawRunner.from_config(EvalConfig.from_yaml(config_path))

    @pytest.mark.parametrize("effort", list(OPENCLAW_EFFORTS))
    def test_valid_efforts(self, effort):
        runner = OpenClawRunner(effort=effort)
        assert runner._effort == effort

    def test_execute_prompt_mode(self, tmp_path, monkeypatch):
        captured = {}
        
        class FakeProcess:
            pid = 1234
            returncode = 0
            
            def communicate(self, input=None, timeout=None):
                captured["input"] = input
                return (json.dumps({
                    "ok": True,
                    "costUsd": 0.01,
                    "usage": {"input": 10, "output": 5},
                }).encode(), b"")
        
        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen", fake_popen)
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        result = runner.execute(None, "What is 2+2?", workspace, "claude-sonnet")
        
        assert result.exit_code == 0
        assert result.cost_usd == 0.01
        assert captured["input"] is None
        assert captured["command"][-1] == "What is 2+2?"
        assert "openclaw" in captured["command"]
        assert "--model" in captured["command"]
        assert "claude-sonnet" in captured["command"]

    def test_execute_skill_mode(self, tmp_path, monkeypatch):
        captured = {}
        
        class FakeProcess:
            pid = 1234
            returncode = 0
            
            def communicate(self, input=None, timeout=None):
                captured["input"] = input
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: (captured.update(command=cmd), FakeProcess())[1])
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        runner.execute("my-skill", "arg1 arg2", workspace, "model")
        
        assert captured["input"] is None
        assert captured["command"][-1] == "/my-skill arg1 arg2"

    def test_execute_skill_mode_no_args(self, tmp_path, monkeypatch):
        captured = {}
        
        class FakeProcess:
            pid = 1234
            returncode = 0
            
            def communicate(self, input=None, timeout=None):
                captured["input"] = input
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: (captured.update(input=None), FakeProcess())[1])
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        
        class FP:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                captured["input"] = input
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: (captured.update(command=cmd), FP())[1])
        
        runner.execute("my-skill", "", workspace, "model")
        assert captured["input"] is None
        assert captured["command"][-1] == "/my-skill"

    def test_warns_on_non_default_budget(self, tmp_path, monkeypatch):
        class FakeProcess:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        
        with pytest.warns(RuntimeWarning, match="does not enforce"):
            runner.execute(None, "p", workspace, "m", max_budget_usd=25.0)

    def test_no_warning_on_default_budget(self, tmp_path, monkeypatch):
        class FakeProcess:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runner.execute(None, "p", workspace, "m", max_budget_usd=5.0)

    def test_warns_on_system_prompt(self, tmp_path, monkeypatch):
        class FakeProcess:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                return b'{"ok": true}', b""
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        
        with pytest.warns(RuntimeWarning, match="ignores system_prompt"):
            runner.execute(
                None, "p", workspace, "m", system_prompt="Be helpful")

    def test_timeout_kills_process_group(self, tmp_path, monkeypatch):
        class FakeProcess:
            pid = 9999
            returncode = -9
            
            def communicate(self, input=None, timeout=None):
                raise subprocess.TimeoutExpired("openclaw", timeout)
            
            def wait(self):
                pass
        
        killed = []
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.os.killpg",
            lambda pid, sig: killed.append((pid, sig)))
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        result = runner.execute(None, "p", workspace, "m", timeout_s=5)
        
        assert killed and killed[0][0] == 9999
        assert result.exit_code == 2
        assert "Timed out" in result.stderr

    def test_interrupt_kills_process_group(self, tmp_path, monkeypatch):
        class FakeProcess:
            pid = 8888
            
            def communicate(self, input=None, timeout=None):
                raise KeyboardInterrupt
        
        killed = []
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen",
            lambda cmd, **kw: FakeProcess())
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.os.killpg",
            lambda pid, sig: killed.append((pid, sig)))
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner()
        
        with pytest.raises(KeyboardInterrupt):
            runner.execute(None, "p", workspace, "m")
        
        assert killed and killed[0][0] == 8888

    def test_env_merging(self, tmp_path, monkeypatch):
        captured = {}
        
        class FakeProcess:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                return b'{"ok": true}', b""
        
        def fake_popen(cmd, **kw):
            captured["env"] = kw.get("env", {})
            return FakeProcess()
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen", fake_popen)
        monkeypatch.setenv("EXISTING", "from_os")
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner(env={"RUNNER_VAR": "from_runner"})
        runner.execute(
            None, "p", workspace, "m",
            extra_env={"EXTRA_VAR": "from_extra"})
        
        assert captured["env"]["EXISTING"] == "from_os"
        assert captured["env"]["RUNNER_VAR"] == "from_runner"
        assert captured["env"]["EXTRA_VAR"] == "from_extra"

    def test_effort_passed_to_command(self, tmp_path, monkeypatch):
        captured = {}
        
        class FakeProcess:
            pid = 1234
            returncode = 0
            def communicate(self, input=None, timeout=None):
                return b'{"ok": true}', b""
        
        def fake_popen(cmd, **kw):
            captured["command"] = cmd
            return FakeProcess()
        
        monkeypatch.setattr(
            "agent_eval.agent.openclaw.subprocess.Popen", fake_popen)
        
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        runner = OpenClawRunner(effort="high")
        runner.execute(None, "p", workspace, "m")
        
        cmd = captured["command"]
        assert "--thinking" in cmd
        idx = cmd.index("--thinking")
        assert cmd[idx + 1] == "high"
