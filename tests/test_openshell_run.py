"""Tests for the OpenShell backend orchestrator."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import agent_eval._bootstrap
from agent_eval.openshell.run import (
    _M365_FORWARD_ENV,
    _M365_GRAPH_CURL_PATH,
    _M365_HEADER_PATH,
    _child_env,
    _ensure_m365_credentials,
    _install_m365_file_auth,
    _m365_usable,
    _openai_compat_base_url,
    _resolve_prompt,
    _sandbox_env,
    _stage_forge_ai_gateway_ca,
    _setup_scene,
    build_openclaw_eval_config,
    qualify_openclaw_model,
)


@pytest.mark.parametrize("code,expected", [(0, True), (3, False), (2, True), (127, True)])
def test_optional_openclaw_output_probe(code, expected):
    import asyncio
    from agent_eval.openshell.run import _openclaw_output_present

    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=code)))
    assert asyncio.run(_openclaw_output_present(sandbox, "test-sandbox")) is expected
    assert sandbox.exec.call_args.args[0] == "test-sandbox"


def test_llm_preflight_resolves_environment_reference_in_memory(tmp_path, monkeypatch):
    import asyncio
    import shutil
    import subprocess
    from agent_eval.openshell.run import _run_openclaw_llm_preflight

    node = shutil.which("node")
    if not node:
        pytest.skip("node required for preflight script regression")
    path = tmp_path / "config.json"
    original = json.dumps({"models": {"providers": {"inference": {
        "baseUrl": "https://example.invalid/v1", "apiKey": "${PREFLIGHT_TEST_KEY}",
    }}}})
    path.write_text(original)
    monkeypatch.setenv("PREFLIGHT_TEST_KEY", "synthetic-test-value")
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=0, stdout="", stderr="")))
    asyncio.run(_run_openclaw_llm_preflight(sandbox, "test", path, "inference/test"))
    argv = sandbox.exec.call_args.args[1]
    stub = (
        "global.fetch=async(url,opts)=>{"
        "const body=JSON.parse(opts.body);"
        "if(body.max_tokens!==512)throw new Error('insufficient preflight budget');"
        "if(opts.headers.authorization!=='Bearer synthetic-test-value')throw new Error('bad auth');"
        "return {ok:true,status:200,text:async()=>JSON.stringify({choices:[{message:{content:'OK'}}]})};};"
    )
    result = subprocess.run([node, "-e", stub + argv[2], *argv[3:]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "LLM_PREFLIGHT_OK" in result.stdout
    assert "synthetic-test-value" not in result.stdout
    assert path.read_text() == original


@pytest.mark.parametrize("code,error", [
    (124, "timeout"),
    (1, "LLM_PREFLIGHT_FAILED This operation was aborted"),
    (1, "LLM_PREFLIGHT_FAILED HTTP 504 upstream timeout"),
])
def test_preflight_retries_one_transient_failure(code, error):
    import asyncio
    from agent_eval.openshell.run import _run_openclaw_llm_preflight
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[
        SimpleNamespace(return_code=code, stdout="", stderr=error),
        SimpleNamespace(return_code=0, stdout="LLM_PREFLIGHT_OK", stderr=""),
    ]))
    with patch("agent_eval.openshell.run.asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_openclaw_llm_preflight(sandbox, "probe", Path("config"), "inference/glm"))
    assert sandbox.exec.call_count == 2


@pytest.mark.parametrize("error,count", [
    ("LLM_PREFLIGHT_FAILED HTTP 401 unauthorized", 1),
    ("LLM_PREFLIGHT_FAILED HTTP 400 invalid model", 1),
    ("LLM_PREFLIGHT_FAILED empty model response", 1),
    ("LLM_PREFLIGHT_FAILED HTTP 503 unavailable", 2),
])
def test_preflight_retry_is_bounded_and_not_for_auth_or_config(error, count):
    import asyncio
    from agent_eval.openshell.run import _run_openclaw_llm_preflight
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
        return_code=1, stdout="", stderr=error)))
    with patch("agent_eval.openshell.run.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(RuntimeError, match="preflight failed"):
            asyncio.run(_run_openclaw_llm_preflight(sandbox, "probe", Path("config"), "inference/glm"))
    assert sandbox.exec.call_count == count


class TestChildEnv:
    """Tests for _child_env (bootstrap sentinel stripping)."""

    def test_strips_bootstrap_sentinel(self, monkeypatch):
        sentinel = agent_eval._bootstrap._SENTINEL
        monkeypatch.setenv(sentinel, "1")
        
        env = _child_env()
        
        assert sentinel not in env

    def test_strips_sentinel_after_extras(self, monkeypatch):
        sentinel = agent_eval._bootstrap._SENTINEL
        monkeypatch.setenv(sentinel, "1")
        
        # Even if extra tries to reinstate it, it's stripped
        env = _child_env({sentinel: "should-be-stripped"})
        
        assert sentinel not in env

    def test_preserves_other_env_vars(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("PATH", "/usr/bin")
        
        env = _child_env()
        
        assert env["ANTHROPIC_API_KEY"] == "test-key"
        assert "PATH" in env

    def test_includes_extra_vars(self, monkeypatch):
        env = _child_env({"CUSTOM_VAR": "custom-value"})
        
        assert env["CUSTOM_VAR"] == "custom-value"

    def test_extra_overrides_existing(self, monkeypatch):
        monkeypatch.setenv("EXISTING", "original")
        
        env = _child_env({"EXISTING": "overridden"})
        
        assert env["EXISTING"] == "overridden"


class TestSandboxEnv:
    """Tests for _sandbox_env (API key forwarding)."""

    def test_forwards_api_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-456")
        
        config = _mock_config()
        env = _sandbox_env(config)
        
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-123"
        assert env["OPENAI_API_KEY"] == "sk-openai-456"

    def test_forwards_provider_config(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.api")
        
        config = _mock_config()
        env = _sandbox_env(config)
        
        assert env["ANTHROPIC_MODEL"] == "claude-3"
        assert env["ANTHROPIC_BASE_URL"] == "https://custom.api"

    def test_merges_execution_env(self, monkeypatch):
        monkeypatch.setenv("SOURCE_VAR", "resolved-value")
        
        config = _mock_config(
            execution_env={"CUSTOM": "direct", "RESOLVED": "$SOURCE_VAR"}
        )
        env = _sandbox_env(config)
        
        assert env["CUSTOM"] == "direct"
        assert env["RESOLVED"] == "resolved-value"

    def test_merges_runner_env(self, monkeypatch):
        config = _mock_config(runner_env={"RUNNER_VAR": "runner-value"})
        env = _sandbox_env(config)
        
        assert env["RUNNER_VAR"] == "runner-value"

    def test_runner_env_overrides_execution_env(self, monkeypatch):
        config = _mock_config(
            execution_env={"SHARED": "from-execution"},
            runner_env={"SHARED": "from-runner"},
        )
        env = _sandbox_env(config)
        
        assert env["SHARED"] == "from-runner"

    def test_skips_none_values(self, monkeypatch):
        config = _mock_config(execution_env={"NULL_VAR": None})
        env = _sandbox_env(config)
        
        assert "NULL_VAR" not in env

    def test_skips_unresolved_refs(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        config = _mock_config(execution_env={"UNRESOLVED": "$MISSING_VAR"})
        env = _sandbox_env(config)
        
        assert "UNRESOLVED" not in env

    def test_forwards_m365_from_allowlist(self, monkeypatch):
        monkeypatch.setenv("M365_ACCESS_TOKEN", "tok-allowlist")
        monkeypatch.setenv("M365_USER", "tbx-demo2@dev.mscloud.ibm.com")
        monkeypatch.setenv("M365_TENANT_ID", "tenant-id")
        monkeypatch.setenv("M365_CLIENT_ID", "client-id")
        monkeypatch.setenv("M365_CLIENT_SECRET", "client-secret")

        env = _sandbox_env(_mock_config())

        assert env["M365_ACCESS_TOKEN"] == "tok-allowlist"
        assert env["M365_USER"] == "tbx-demo2@dev.mscloud.ibm.com"
        assert env["M365_TENANT_ID"] == "tenant-id"
        assert env["M365_CLIENT_ID"] == "client-id"
        assert env["M365_CLIENT_SECRET"] == "client-secret"
        assert set(_M365_FORWARD_ENV) <= set(env)

    def test_forwards_m365_from_execution_env(self, monkeypatch):
        monkeypatch.setenv("M365_ACCESS_TOKEN", "tok-exec")
        monkeypatch.setenv("M365_USER", "user@example.com")
        monkeypatch.delenv("M365_TENANT_ID", raising=False)

        config = _mock_config(
            execution_env={
                "M365_ACCESS_TOKEN": "$M365_ACCESS_TOKEN",
                "M365_USER": "$M365_USER",
                "M365_TENANT_ID": "$M365_TENANT_ID",
                "FORGE_SOURCES": "m365-only",
            }
        )
        env = _sandbox_env(config)

        assert env["M365_ACCESS_TOKEN"] == "tok-exec"
        assert env["M365_USER"] == "user@example.com"
        assert "M365_TENANT_ID" not in env
        assert env["FORGE_SOURCES"] == "m365-only"


class TestForgeAiGatewayCa:
    """Tests for the CI-scoped Node trust bundle staging."""

    def test_stages_public_ca_and_sets_node_trust(self, tmp_path, monkeypatch):
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        source = tmp_path / "ca.crt"
        source.write_text("-----BEGIN CERTIFICATE-----\npublic-ca\n-----END CERTIFICATE-----\n")
        monkeypatch.setenv("AGENT_EVAL_FORGE_AI_GATEWAY_CA_FILE", str(source))
        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.exec = AsyncMock(return_value=MagicMock(return_code=0))
        sandbox.upload = AsyncMock()
        env = {}

        asyncio.run(_stage_forge_ai_gateway_ca(sandbox, "case-001", env))

        sandbox.upload.assert_not_awaited()
        assert env["NODE_EXTRA_CA_CERTS"] == "/sandbox/ca.crt"


class TestResolvePrompt:
    """Tests for _resolve_prompt (template resolution)."""

    def test_literal_prompt(self):
        config = _mock_config(prompt="What is 2+2?")
        result = _resolve_prompt(config, {})
        
        assert result == "What is 2+2?"

    def test_simple_format_template(self):
        config = _mock_config(prompt="Process {ticket_id}")
        result = _resolve_prompt(config, {"ticket_id": "JIRA-123"})
        
        assert result == "Process JIRA-123"

    def test_optional_field_present(self):
        config = _mock_config(prompt="Do {task} with {option?}")
        result = _resolve_prompt(config, {"task": "work", "option": "extra"})
        
        assert result == "Do work with extra"

    def test_optional_field_missing(self):
        config = _mock_config(prompt="Do {task} with {option?}")
        result = _resolve_prompt(config, {"task": "work"})
        
        assert result == "Do work with"

    def test_missing_required_field_raises(self):
        config = _mock_config(prompt="Process {required_field}")
        
        with pytest.raises(ValueError, match="Missing required field"):
            _resolve_prompt(config, {})

    def test_jinja_template(self):
        config = _mock_config(prompt="{{ input.prompt }}")
        result = _resolve_prompt(config, {"prompt": "Hello from Jinja"})
        
        assert result == "Hello from Jinja"

    def test_jinja_with_default_filter(self):
        config = _mock_config(prompt="{{ input.name | default('Anonymous') }}")
        result = _resolve_prompt(config, {})
        
        assert result == "Anonymous"

    def test_jinja_missing_required_raises(self):
        config = _mock_config(prompt="{{ input.missing_field }}")
        
        with pytest.raises(ValueError, match="Undefined variable"):
            _resolve_prompt(config, {})

    def test_fallback_to_input_prompt(self):
        config = _mock_config(prompt=None, arguments=None)
        result = _resolve_prompt(config, {"prompt": "Fallback prompt"})
        
        assert result == "Fallback prompt"

    def test_empty_when_no_template_or_input(self):
        config = _mock_config(prompt=None, arguments=None)
        result = _resolve_prompt(config, {})
        
        assert result == ""

    def test_uses_arguments_if_no_prompt(self):
        config = _mock_config(prompt=None, arguments="Run {task}")
        result = _resolve_prompt(config, {"task": "eval"})
        
        assert result == "Run eval"


class TestRunCaseEnvForwarding:
    """Tests verifying _run_case passes env to sandbox.exec()."""

    def test_diagnostic_helpers_keep_only_assistant_text_and_redact_secret(self, monkeypatch, tmp_path):
        from agent_eval.openshell.run import _last_assistant_text, _write_failure

        monkeypatch.setenv("TEST_API_TOKEN", "secret-value-123")
        assert _last_assistant_text([
            {"type": "assistant", "text": "First thought"},
            {"type": "tool", "text": "tool output"},
            {"type": "assistant", "text": "Partial recommendation"},
        ]) == "Partial recommendation"
        _write_failure(tmp_path, "preflight", "Bearer abc123 secret-value-123", 124)
        failure = json.loads((tmp_path / "failure.json").read_text())
        assert failure["exit_code"] == 124
        assert "secret-value-123" not in failure["error"]
        assert "abc123" not in failure["error"]

    def test_create_failure_writes_diagnostic(self, tmp_path):
        from agent_eval.openshell.run import _run_case
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        case = tmp_path / "cases" / "early-failure"
        case.mkdir(parents=True)
        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.create = AsyncMock(side_effect=RuntimeError("image pull refused"))
        sandbox.delete = AsyncMock()
        with pytest.raises(RuntimeError, match="image pull refused"):
            asyncio.run(_run_case(sandbox, _mock_config(prompt="test"), case,
                                  "model", "image:v1", tmp_path / "runs",
                                  asyncio.Semaphore(1), keep=False, scene_active=True))
        failure = json.loads((tmp_path / "runs/cases/early-failure/failure.json").read_text())
        assert failure["phase"] == "sandbox-create"
        assert "image pull refused" in failure["error"]

    def test_agent_failure_preserves_partial_text_only_as_diagnostic(self, tmp_path, monkeypatch):
        from agent_eval.openshell.run import _run_case
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        case = tmp_path / "cases" / "agent-failure"
        case.mkdir(parents=True)
        (case / "input.yaml").write_text("prompt: test\n")
        config = _mock_config(prompt="test")
        config.runner.type = "openclaw"
        config.runner.providers = None
        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.create = AsyncMock()
        sandbox.upload = AsyncMock()
        sandbox.download = AsyncMock()
        sandbox.delete = AsyncMock()
        error_envelope = json.dumps({"ok": False, "status": "error", "final": "",
            "error": {"message": "malformed tool call"},
            "payloads": [{"text": "malformed tool call", "isError": True}]})

        async def fake_exec(_name, command, **_kwargs):
            if command[:2] == ["sh", "-c"] and command[2].startswith("test -e "):
                return SimpleNamespace(return_code=1, stdout="", stderr="")
            if command and command[0] == "openclaw":
                return SimpleNamespace(return_code=1, stdout=error_envelope, stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        sandbox.exec = AsyncMock(side_effect=fake_exec)

        async def fake_harvest(_sandbox, _name, *, case_output, **_kwargs):
            (case_output / "openclaw-trajectory-events.jsonl").write_text("observed transcript")
            return [{"type": "assistant", "text": "I found one blocker, but have not finished."}]
        monkeypatch.setattr("agent_eval.openshell.run._harvest_openclaw_events", fake_harvest)

        result = asyncio.run(_run_case(sandbox, config, case, "model", "image:v1",
                                   tmp_path / "runs", asyncio.Semaphore(1),
                                   keep=False, scene_active=True))
        case_output = tmp_path / "runs/cases/agent-failure"
        assert result["exit_code"] == 1
        assert (case / "output/response.txt").read_text() == ""
        assert (case_output / "agent-response.txt").read_text() == "I found one blocker, but have not finished."
        failure = json.loads((case_output / "failure.json").read_text())
        assert failure["phase"] == "agent-exec"
        assert "malformed tool call" in failure["error"]

    def test_run_case_passes_env_to_exec(self, tmp_path):
        """Verify sandbox.exec receives forwarded env vars."""
        from agent_eval.openshell.run import _run_case
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio
        
        # Set up env vars that should be forwarded
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
        
        try:
            config = _mock_config(prompt="test")
            
            # Create staged case directory
            staged_case = tmp_path / "cases" / "case-001"
            staged_case.mkdir(parents=True)
            (staged_case / "input.yaml").write_text(yaml.safe_dump({}))
            # The local staging repo is not agent input and may conflict with
            # an image-owned .git file (e.g. a packaged workspace checkout).
            (staged_case / ".git" / "hooks").mkdir(parents=True)
            (staged_case / ".git" / "hooks" / "sample").write_text("local-only")
            
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            
            # Mock sandbox
            sandbox = MagicMock(spec=OpenShellSandbox)
            sandbox.create = AsyncMock()
            sandbox.upload = AsyncMock()
            sandbox.download = AsyncMock()
            sandbox.delete = AsyncMock()
            
            exec_result = MagicMock()
            exec_result.stdout = json.dumps({"ok": True})
            exec_result.stderr = ""
            exec_result.return_code = 0
            async def fake_exec(_name, command, **kwargs):
                if command[:2] == ["sh", "-c"] and command[2].startswith("test -e "):
                    return SimpleNamespace(return_code=1, stdout="", stderr="")
                return exec_result
            sandbox.exec = AsyncMock(side_effect=fake_exec)
            
            async def run_test():
                sem = asyncio.Semaphore(1)
                await _run_case(
                    sandbox, config, staged_case, "model", "image:v1",
                    output_dir, sem, keep=False, scene_active=True,
                )
            
            asyncio.run(run_test())
            
            # Verify exec was called with env
            sandbox.exec.assert_called()
            call_kwargs = sandbox.exec.call_args[1]
            assert "env" in call_kwargs
            assert call_kwargs["env"]["ANTHROPIC_API_KEY"] == "test-api-key"
            sandbox.upload.assert_awaited_once_with(
                sandbox.create.call_args.args[0],
                staged_case / "input.yaml",
                "/sandbox",
            )
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_run_case_forwards_m365_into_exec_env(self, tmp_path, monkeypatch):
        from agent_eval.openshell.run import _run_case
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        monkeypatch.setenv("M365_ACCESS_TOKEN", "tok-sandbox")
        monkeypatch.setenv("M365_USER", "tbx-demo2@dev.mscloud.ibm.com")
        monkeypatch.setenv("M365_TENANT_ID", "tenant")
        monkeypatch.setenv("M365_CLIENT_ID", "client")
        monkeypatch.setenv("M365_CLIENT_SECRET", "secret")

        config = _mock_config(
            prompt="brief me",
            execution_env={
                "M365_ACCESS_TOKEN": "$M365_ACCESS_TOKEN",
                "M365_USER": "$M365_USER",
                "M365_TENANT_ID": "$M365_TENANT_ID",
                "M365_CLIENT_ID": "$M365_CLIENT_ID",
                "M365_CLIENT_SECRET": "$M365_CLIENT_SECRET",
                "FORGE_SOURCES": "m365-only",
            },
        )
        config.runner.type = "openclaw"
        config.runner.providers = None

        staged_case = tmp_path / "cases" / "case-001"
        staged_case.mkdir(parents=True)
        (staged_case / "input.yaml").write_text(yaml.safe_dump({}))
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.create = AsyncMock()
        sandbox.upload = AsyncMock()
        sandbox.download = AsyncMock()
        sandbox.delete = AsyncMock()
        exec_result = MagicMock()
        exec_result.stdout = json.dumps({"ok": True})
        exec_result.stderr = ""
        exec_result.return_code = 0
        sandbox.exec = AsyncMock(return_value=exec_result)

        async def run_test():
            sem = asyncio.Semaphore(1)
            await _run_case(
                sandbox, config, staged_case, "model", "image:v1",
                output_dir, sem, keep=False, scene_active=True,
            )

        asyncio.run(run_test())

        agent_calls = [
            c for c in sandbox.exec.call_args_list
            if c.kwargs.get("env") and c.kwargs["env"].get("M365_ACCESS_TOKEN")
        ]
        assert agent_calls, "sandbox.exec was not called with M365 tokens in env"
        env = agent_calls[-1].kwargs["env"]
        assert env["M365_ACCESS_TOKEN"] == "tok-sandbox"
        assert env["M365_USER"] == "tbx-demo2@dev.mscloud.ibm.com"
        assert env["M365_TENANT_ID"] == "tenant"
        assert env["M365_CLIENT_ID"] == "client"
        assert env["M365_CLIENT_SECRET"] == "secret"
        assert env["M365_AUTH_HEADER_FILE"] == _M365_HEADER_PATH
        assert env["M365_GRAPH_CURL"] == _M365_GRAPH_CURL_PATH
        assert env["FORGE_SOURCES"] == "m365-only"


class TestRunCasePromptResolution:
    """Tests verifying _run_case resolves prompt templates."""

    def test_run_case_resolves_jinja_prompt(self, tmp_path):
        """Verify Jinja templates are resolved before sending to sandbox."""
        from agent_eval.openshell.run import _run_case
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio
        
        config = _mock_config(prompt="{{ input.message }}")
        
        staged_case = tmp_path / "cases" / "case-001"
        staged_case.mkdir(parents=True)
        (staged_case / "input.yaml").write_text(
            yaml.safe_dump({"message": "Hello resolved"})
        )
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.create = AsyncMock()
        sandbox.upload = AsyncMock()
        sandbox.download = AsyncMock()
        sandbox.delete = AsyncMock()
        
        exec_result = MagicMock()
        exec_result.stdout = json.dumps({"ok": True})
        exec_result.stderr = ""
        exec_result.return_code = 0
        sandbox.exec = AsyncMock(return_value=exec_result)
        
        async def run_test():
            sem = asyncio.Semaphore(1)
            await _run_case(
                sandbox, config, staged_case, "model", "image:v1",
                output_dir, sem, keep=False, scene_active=True,
            )
        
        asyncio.run(run_test())
        
        # Verify stdin contains resolved prompt
        call_kwargs = sandbox.exec.call_args[1]
        stdin_data = call_kwargs.get("stdin", b"")
        assert b"Hello resolved" in stdin_data


class TestInstallM365FileAuth:
    """Graph header-file helpers installed inside the sandbox."""

    def test_noop_without_token(self):
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        sandbox = MagicMock(spec=OpenShellSandbox)
        sandbox.exec = AsyncMock()
        env = {"M365_USER": "user@example.com"}

        asyncio.run(_install_m365_file_auth(sandbox, "sbx", env))

        sandbox.exec.assert_not_called()
        assert "M365_AUTH_HEADER_FILE" not in env

    def test_writes_header_and_wrapper(self):
        from agent_eval.openshell.sandbox import OpenShellSandbox
        import asyncio

        sandbox = MagicMock(spec=OpenShellSandbox)
        ok = MagicMock()
        ok.return_code = 0
        sandbox.exec = AsyncMock(return_value=ok)
        env = {
            "M365_ACCESS_TOKEN": "eyJ-test-token",
            "M365_USER": "user@example.com",
        }

        asyncio.run(_install_m365_file_auth(sandbox, "sbx", env))

        assert env["M365_AUTH_HEADER_FILE"] == _M365_HEADER_PATH
        assert env["M365_GRAPH_CURL"] == _M365_GRAPH_CURL_PATH
        commands = [call.args[1] for call in sandbox.exec.call_args_list]
        assert ["tee", _M365_HEADER_PATH] in commands
        header_call = next(
            c for c in sandbox.exec.call_args_list if c.args[1][:2] == ["tee", _M365_HEADER_PATH]
        )
        assert b"Authorization: Bearer eyJ-test-token" in header_call.kwargs["stdin"]


class TestEnsureM365Credentials:
    """Forge Graph runs must fail closed when orchestrator M365_* is empty."""

    def test_placeholder_not_usable(self):
        assert _m365_usable(None) is False
        assert _m365_usable("") is False
        assert _m365_usable("<replace-with-graph-token>") is False
        assert _m365_usable("eyJ-real-token") is True

    def test_skips_when_no_m365_declared(self, monkeypatch):
        monkeypatch.delenv("M365_ACCESS_TOKEN", raising=False)
        config = _mock_config()
        config.config_path = None
        _ensure_m365_credentials(config)

    def test_raises_when_execution_env_unresolved(self, monkeypatch):
        monkeypatch.delenv("M365_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("M365_USER", raising=False)
        config = _mock_config(
            execution_env={
                "M365_ACCESS_TOKEN": "$M365_ACCESS_TOKEN",
                "M365_USER": "$M365_USER",
            }
        )
        config.config_path = None
        with pytest.raises(RuntimeError, match="M365 Graph credentials"):
            _ensure_m365_credentials(config)

    def test_raises_on_placeholder_token(self, monkeypatch):
        monkeypatch.setenv("M365_ACCESS_TOKEN", "<replace-with-graph-token>")
        monkeypatch.setenv("M365_USER", "user@example.com")
        config = _mock_config(
            execution_env={"M365_ACCESS_TOKEN": "$M365_ACCESS_TOKEN"}
        )
        config.config_path = None
        with pytest.raises(RuntimeError, match="placeholder"):
            _ensure_m365_credentials(config)

    def test_ok_when_token_and_user_set(self, monkeypatch):
        monkeypatch.setenv("M365_ACCESS_TOKEN", "tok-live")
        monkeypatch.setenv("M365_USER", "user@example.com")
        config = _mock_config(
            execution_env={
                "M365_ACCESS_TOKEN": "$M365_ACCESS_TOKEN",
                "M365_USER": "$M365_USER",
            }
        )
        config.config_path = None
        _ensure_m365_credentials(config)

    def test_scene_user_fills_m365_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("M365_ACCESS_TOKEN", "tok-live")
        monkeypatch.delenv("M365_USER", raising=False)
        eval_yaml = tmp_path / "eval.yaml"
        scenes = tmp_path / "scenes"
        scenes.mkdir()
        eval_yaml.write_text("scene: monday-acquisition\n", encoding="utf-8")
        (scenes / "monday-acquisition.yaml").write_text(
            "m365:\n  user: tbx-demo2@dev.mscloud.ibm.com\n  seed: external\n",
            encoding="utf-8",
        )
        config = _mock_config()
        config.config_path = eval_yaml
        _ensure_m365_credentials(config)
        assert os.environ["M365_USER"] == "tbx-demo2@dev.mscloud.ibm.com"

    def test_external_scene_requires_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("M365_ACCESS_TOKEN", raising=False)
        eval_yaml = tmp_path / "eval.yaml"
        scenes = tmp_path / "scenes"
        scenes.mkdir()
        eval_yaml.write_text("scene: monday-acquisition\n", encoding="utf-8")
        (scenes / "monday-acquisition.yaml").write_text(
            "m365:\n  user: tbx-demo2@dev.mscloud.ibm.com\n  seed: external\n",
            encoding="utf-8",
        )
        config = _mock_config()
        config.config_path = eval_yaml
        with pytest.raises(RuntimeError, match="M365_ACCESS_TOKEN"):
            _ensure_m365_credentials(config)


class TestSetupSceneM365:
    """Scene YAML with m365.seed=external must not look like a failed Slack seed."""

    def test_records_external_mailbox_and_skips_mocks(self, tmp_path, caplog, monkeypatch):
        monkeypatch.delenv("M365_ACCESS_TOKEN", raising=False)
        eval_yaml = tmp_path / "eval.yaml"
        scenes = tmp_path / "scenes"
        scenes.mkdir()
        eval_yaml.write_text("scene: monday-acquisition\n", encoding="utf-8")
        (scenes / "monday-acquisition.yaml").write_text(
            "\n".join(
                [
                    "name: ibm-forge-monday-briefing-m365",
                    "slack:",
                    "  enabled: false",
                    "m365:",
                    "  user: tbx-demo2@dev.mscloud.ibm.com",
                    "  seed: external",
                    "crabline_seeds: []",
                    "smolclaw_seeds: []",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        config = MagicMock()
        config.config_path = eval_yaml
        out = tmp_path / "run"
        out.mkdir()

        with caplog.at_level("INFO"):
            assert _setup_scene(config, out) is True

        meta = json.loads((out / "scene-seed.json").read_text(encoding="utf-8"))
        assert meta["m365"]["seed"] == "external"
        assert meta["m365"]["user"] == "tbx-demo2@dev.mscloud.ibm.com"
        assert meta["m365"]["access_token"] == "missing"
        assert meta["m365"]["slack_enabled"] is False
        assert "crabline" not in meta
        assert "smolclaw" not in meta
        text = caplog.text
        assert "m365=external (tbx-demo2@dev.mscloud.ibm.com)" in text
        assert "slack=disabled" in text
        assert "Scene seeded: 0 Slack" not in text


def _mock_config(
    prompt=None,
    arguments=None,
    execution_env=None,
    runner_env=None,
):
    """Create a mock EvalConfig for testing."""
    config = MagicMock()
    config.execution.prompt = prompt
    config.execution.arguments = arguments
    config.execution.timeout = 300
    config.execution.env = execution_env or {}
    config.runner.type = "claude-code"
    config.runner.effort = None
    config.runner.settings = {}
    config.runner.env = runner_env or {}
    config.runner.providers = None
    config.outputs = []
    return config


class TestOpenclawEvalConfig:
    """openclaw-eval.json must use LiteLLM/inference, not anthropic/claude-sonnet."""

    _WXNB_PROVIDERS = {
        "inference": {
            "baseUrl": "https://inference.local/v1",
            "apiKey": "empty",
            "models": [
                {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "api": "openai-completions"}
            ],
        }
    }

    def test_qualifies_litellm_alias_to_inference(self):
        assert (
            qualify_openclaw_model("claude-sonnet", self._WXNB_PROVIDERS)
            == "inference/claude-sonnet"
        )

    def test_keeps_already_qualified_model(self):
        assert (
            qualify_openclaw_model("inference/claude-sonnet", self._WXNB_PROVIDERS)
            == "inference/claude-sonnet"
        )

    def test_wxvnb_mismatch_adds_claude_sonnet_id(self):
        cfg, qualified = build_openclaw_eval_config(
            self._WXNB_PROVIDERS, "claude-sonnet"
        )
        assert qualified == "inference/claude-sonnet"
        assert cfg["agents"]["defaults"]["model"]["primary"] == "inference/claude-sonnet"
        assert "anthropic" not in cfg["models"]["providers"]
        assert cfg["models"]["mode"] == "replace"
        inf = cfg["models"]["providers"]["inference"]
        ids = [m["id"] for m in inf["models"]]
        assert "claude-sonnet" in ids
        assert inf["api"] == "openai-completions"
        assert inf["baseUrl"] == "https://inference.local/v1"

    def test_cluster_litellm_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock")
        providers = {
            "inference": {
                "baseUrl": "http://litellm.ab-eval-flow.svc.cluster.local:4000/v1",
                "api": "openai-completions",
                "apiKey": "$ANTHROPIC_API_KEY",
                "models": [{"id": "claude-sonnet", "name": "Claude Sonnet"}],
            }
        }
        cfg, qualified = build_openclaw_eval_config(providers, "claude-sonnet")
        inf = cfg["models"]["providers"]["inference"]
        assert qualified == "inference/claude-sonnet"
        assert inf["apiKey"] == "mock"
        assert inf["baseUrl"].endswith("/v1")
        assert inf["models"][0]["id"] == "claude-sonnet"

    def test_preserves_declared_model_capabilities(self):
        model = {
            "id": "glm", "name": "GLM", "reasoning": True,
            "maxTokens": 32768, "contextWindow": 131072,
            "input": ["text"], "cost": {"input": 1, "output": 2},
        }
        providers = {"inference": {"baseUrl": "https://model.example/v1", "models": [model]}}
        cfg, _ = build_openclaw_eval_config(providers, "inference/glm")
        actual = cfg["models"]["providers"]["inference"]["models"][0]
        for field, value in model.items():
            assert actual[field] == value
        assert model["maxTokens"] == 32768

    def test_unspecified_model_capabilities_keep_defaults(self):
        providers = {"inference": {"baseUrl": "https://model.example/v1", "models": [{"id": "plain"}]}}
        cfg, _ = build_openclaw_eval_config(providers, "inference/plain")
        actual = cfg["models"]["providers"]["inference"]["models"][0]
        assert actual["maxTokens"] == 8192
        assert actual["reasoning"] is False

    def test_preserves_private_network_provider_opt_in(self):
        providers = {
            "forge-ai-gateway": {
                "baseUrl": "https://host.containers.internal:18084/v1",
                "apiKey": "openshell:placeholder",
                "request": {"allowPrivateNetwork": True},
                "models": [{"id": "rits/zai-org/glm-5-3"}],
            }
        }
        cfg, _ = build_openclaw_eval_config(providers, "rits/zai-org/glm-5-3")
        assert cfg["models"]["providers"]["forge-ai-gateway"]["request"] == {
            "allowPrivateNetwork": True
        }

    def test_appends_v1_to_litellm_base_without_path(self, monkeypatch):
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", "http://litellm.ab-eval-flow.svc:4000"
        )
        providers = {
            "inference": {
                "baseUrl": "$ANTHROPIC_BASE_URL",
                "apiKey": "empty",
                "models": [{"id": "claude-sonnet"}],
            }
        }
        cfg, _ = build_openclaw_eval_config(providers, "claude-sonnet")
        assert (
            cfg["models"]["providers"]["inference"]["baseUrl"]
            == "http://litellm.ab-eval-flow.svc:4000/v1"
        )

    def test_openai_compat_base_url_idempotent(self):
        assert _openai_compat_base_url("http://x:4000/v1") == "http://x:4000/v1"
        assert _openai_compat_base_url("http://x:4000") == "http://x:4000/v1"
