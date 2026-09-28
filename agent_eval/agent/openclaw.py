"""OpenClaw CLI runner implementation.

OpenClaw is an open-source AI coding assistant. This runner invokes it via
`openclaw agent exec --json` for local evaluations. The shared helpers
(build_openclaw_argv, parse_openclaw_*) are also used by the OpenShell backend
for in-sandbox execution.
"""

import json
import logging
import os
import signal
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import EvalRunner, RunResult

logger = logging.getLogger(__name__)

# Valid effort levels for --thinking flag
OPENCLAW_EFFORTS = frozenset({"off", "minimal", "low", "medium", "high"})


def build_openclaw_argv(
    model: str,
    cwd: Optional[Path] = None,
    timeout_s: Optional[int] = None,
    effort: Optional[str] = None,
    auth_env_only: bool = True,
    state_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> List[str]:
    """Build openclaw agent exec argv.

    Shared by OpenClawRunner (local) and openshell backend (in-sandbox).

    Args:
        model: Model identifier (e.g. "inference/claude-sonnet-4").
        cwd: Working directory for the agent.
        timeout_s: Timeout in seconds.
        effort: Thinking effort level (off|minimal|low|medium|high).
        auth_env_only: If True, use --auth-env-only (env keys only; skips
            config entirely). Incompatible with ``config_path`` — OpenClaw
            rejects pairing ``--auth-env-only`` with ``--config``.
        state_dir: Existing state directory retained across the run.
        session_id: Unused (exec mode doesn't use session IDs).
        config_path: Path to openclaw.json (``--config``). Use this for
            custom ``models.providers`` (e.g. inference.local); must set
            ``auth_env_only=False``.

    Returns:
        Command argv list for subprocess. Caller should append the prompt
        as a positional argument.
    """
    if config_path is not None and auth_env_only:
        raise ValueError(
            "openclaw rejects --auth-env-only with --config; "
            "pass auth_env_only=False when using config_path"
        )
    # Use 'agent exec' for isolated headless runs (new format since 2026.7.x)
    cmd = ["openclaw", "agent", "exec", "--json"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if model:
        cmd.extend(["--model", model])
    if timeout_s:
        cmd.extend(["--timeout", str(timeout_s)])
    if effort and effort in OPENCLAW_EFFORTS:
        cmd.extend(["--thinking", effort])
    if cwd:
        cmd.extend(["--cwd", str(cwd)])
    if state_dir is not None:
        cmd.extend(["--state-dir", str(state_dir)])
    if auth_env_only:
        cmd.append("--auth-env-only")
    # Note: prompt is now a positional argument, added by caller
    return cmd


def _parse_openclaw_envelope(
    stdout: bytes, stderr: bytes
) -> Tuple[Optional[dict], str]:
    """Parse OpenClaw JSON envelope.

    OpenClaw with --json outputs a JSON object, but may have log lines before/after.
    We extract the JSON by finding the first '{' and matching closing '}'.

    Returns:
        Tuple of (parsed_data, error_message). If JSON parsing fails,
        parsed_data is None and error_message contains stderr.
    """
    stdout_str = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    stderr_str = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    
    # Try direct JSON parse first
    try:
        data = json.loads(stdout_str)
        if not isinstance(data, dict):
            return None, stderr_str + "OpenClaw stdout was not a JSON object"
        error_msg = data.get("error", {}).get("message") or ""
        return data, stderr_str + error_msg
    except json.JSONDecodeError:
        pass
    
    # Extract JSON object from mixed output (log lines + JSON)
    try:
        start = stdout_str.find('{')
        if start == -1:
            return None, stderr_str
        
        # Find matching closing brace
        depth = 0
        end = start
        for i, c in enumerate(stdout_str[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        json_str = stdout_str[start:end]
        data = json.loads(json_str)
        error_msg = data.get("error", {}).get("message") or ""
        return data, stderr_str + error_msg
    except (json.JSONDecodeError, ValueError):
        return None, stderr_str


def parse_openclaw_to_run_result(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
    duration_s: float,
) -> RunResult:
    """Parse OpenClaw JSON into RunResult (for local runner / EvalHub).

    Args:
        stdout: Raw stdout bytes from openclaw process.
        stderr: Raw stderr bytes from openclaw process.
        returncode: Process exit code.
        duration_s: Execution duration in seconds.

    Returns:
        RunResult with parsed metrics.
    """
    data, error_msg = _parse_openclaw_envelope(stdout, stderr)
    if data is None:
        return RunResult(
            exit_code=returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=error_msg,
            duration_s=round(duration_s, 1),
        )

    return RunResult(
        exit_code=returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=error_msg,
        duration_s=round(duration_s, 1),
        cost_usd=data.get("costUsd"),
        token_usage={
            "input": data.get("usage", {}).get("input", 0),
            "output": data.get("usage", {}).get("output", 0),
        },
        num_turns=data.get("assistantTurns"),
        resolved_model=data.get("model"),
        raw_output=data,
    )


def extract_openclaw_response(stdout: bytes) -> str:
    """Extract the final response text from OpenClaw JSON output.
    
    Args:
        stdout: Raw stdout bytes from openclaw process.
        
    Returns:
        The response text, or empty string if not found.
    """
    data, _ = _parse_openclaw_envelope(stdout, b"")
    if data is None:
        return ""
    if data.get("ok") is False or data.get("status") in ("error", "timeout"):
        return ""
    
    # Try meta.finalAssistantVisibleText first (most reliable)
    meta = data.get("meta", {})
    if meta.get("finalAssistantVisibleText"):
        return meta["finalAssistantVisibleText"]
    
    # Fall back to payloads[0].text
    payloads = data.get("payloads", [])
    if payloads and payloads[0].get("text") and not payloads[0].get("isError"):
        return payloads[0]["text"]
    
    return ""


def parse_openclaw_to_case_dict(
    stdout: bytes,
    stderr: bytes,
    returncode: int,
    duration_s: float,
) -> dict:
    """Parse OpenClaw JSON into case_result dict (for openshell backend per_case).

    Args:
        stdout: Raw stdout bytes from openclaw process.
        stderr: Raw stderr bytes from openclaw process.
        returncode: Process exit code.
        duration_s: Execution duration in seconds.

    Returns:
        Dict with execution metrics for suite run_result.json.
    """
    data, error_msg = _parse_openclaw_envelope(stdout, stderr)
    if data is None:
        return {
            "exit_code": returncode,
            "duration_s": round(duration_s, 1),
            "token_usage": {"input": 0, "output": 0},
            "cost_usd": None,
            "num_turns": None,
            "response_text": "",
            "stderr": error_msg,
        }

    # Gateway-backed `openclaw agent --json` retains its response under
    # `result`; embedded `agent exec` places the same fields at top level.
    if isinstance(data.get("result"), dict):
        data = data["result"]

    # Extract metadata from OpenClaw JSON structure.
    # Quay / agent-exec envelope puts usage/model/turns at the top level;
    # older session envelopes nest them under meta.agentMeta.
    meta = data.get("meta", {})
    agent_meta = meta.get("agentMeta", {})
    context_status = agent_meta.get("contextBudgetStatus", {})

    # Response text
    response_text = meta.get("finalAssistantVisibleText", "") or data.get("final", "")
    if not response_text:
        payloads = data.get("payloads", [])
        if payloads and payloads[0].get("text") and not payloads[0].get("isError"):
            response_text = payloads[0]["text"]
    if data.get("ok") is False or data.get("status") in ("error", "timeout"):
        response_text = ""

    # Token usage: top-level (agent exec) → agentMeta → estimates
    usage = (
        data.get("usage")
        or agent_meta.get("usage")
        or agent_meta.get("lastCallUsage")
        or {}
    )
    if usage:
        input_tokens = usage.get("input", 0) or 0
        output_tokens = usage.get("output", 0) or 0
    else:
        input_tokens = context_status.get("estimatedPromptTokens", 0) or 0
        output_tokens = len(response_text) // 4 if response_text else 0

    resolved_model = (
        data.get("model")
        or agent_meta.get("model")
        or meta.get("executionTrace", {}).get("winnerModel")
    )
    num_turns = data.get("assistantTurns")
    if num_turns is None:
        num_turns = 1

    return {
        "exit_code": returncode,
        "duration_s": round(duration_s, 1),
        "token_usage": {
            "input": input_tokens,
            "output": output_tokens,
        },
        "cost_usd": data.get("costUsd"),
        "num_turns": num_turns,
        "resolved_model": resolved_model,
        "response_text": response_text,
        "stop_reason": meta.get("stopReason") or data.get("status"),
        "stderr": error_msg,
    }


class OpenClawRunner(EvalRunner):
    """Run a skill or prompt with ``openclaw agent exec --json``.

    This runner is used for local evaluations (--runner local) and EvalHub.
    For OpenShell sandboxed execution, use --runner openshell instead.
    """

    @classmethod
    def from_config(cls, config, *, log_prefix=None, **overrides):
        """Construct runner from EvalConfig.

        Args:
            config: EvalConfig instance.
            log_prefix: Progress logging prefix.
            **overrides: Runner-specific overrides (e.g. effort).

        Returns:
            Configured OpenClawRunner instance.
        """
        env = {**config.execution.env, **config.runner.env}
        effort = overrides.get("effort", config.runner.effort)
        if effort and effort not in OPENCLAW_EFFORTS:
            raise ValueError(
                f"Invalid OpenClaw effort '{effort}'. "
                f"Must be one of: {sorted(OPENCLAW_EFFORTS)}"
            )
        return cls(
            effort=effort,
            env=env,
            log_prefix=log_prefix,
        )

    def __init__(
        self,
        effort: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_prefix: Optional[str] = None,
    ):
        """Initialize OpenClawRunner.

        Args:
            effort: Thinking effort level (off|minimal|low|medium|high).
            env: Environment variables to inject.
            log_prefix: Progress logging prefix.
        """
        self._effort = effort
        self._env = env or {}
        self._log_prefix = log_prefix

    @property
    def name(self) -> str:
        return "openclaw"

    def execute(
        self,
        target: Optional[str],
        args: str,
        workspace: Path,
        model: str,
        settings_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
        max_budget_usd: float = 5.0,
        timeout_s: int = 600,
        extra_env: Optional[dict] = None,
    ) -> RunResult:
        """Execute a skill or prompt via OpenClaw.

        Args:
            target: Skill name for skill mode, or None for prompt mode.
            args: Skill arguments or prompt text.
            workspace: Pre-staged workspace directory.
            model: Model identifier.
            settings_path: Unused (OpenClaw has no equivalent).
            system_prompt: System prompt (v1: logs warning and ignores).
            max_budget_usd: Budget limit (v1: logs warning if non-default).
            timeout_s: Timeout in seconds.
            extra_env: Additional environment variables.

        Returns:
            RunResult with execution metrics.
        """
        del settings_path

        if max_budget_usd not in (None, 5.0):
            warnings.warn(
                f"OpenClaw does not enforce max_budget_usd={max_budget_usd}; "
                "timeout is the available cap.",
                RuntimeWarning,
                stacklevel=2,
            )

        if system_prompt:
            warnings.warn(
                "OpenClaw runner ignores system_prompt in v1; "
                "prepend to prompt if needed.",
                RuntimeWarning,
                stacklevel=2,
            )

        workspace = workspace.resolve()

        cmd = build_openclaw_argv(
            model=model,
            cwd=workspace,
            timeout_s=timeout_s,
            effort=self._effort,
            auth_env_only=True,
        )

        if target:
            prompt = f"/{target} {args}" if args else f"/{target}"
        else:
            prompt = args or ""

        # Prompt is now a positional argument in 'agent exec' format
        cmd.append(prompt)

        env = {**os.environ, **self._env, **(extra_env or {})}
        start_time = time.monotonic()

        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            env=env,
        )
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            proc.wait()
            return RunResult(
                exit_code=2,
                stdout="",
                stderr=f"Timed out after {timeout_s}s",
                duration_s=timeout_s,
                resolved_model=model or None,
            )
        except BaseException:
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            raise

        duration_s = time.monotonic() - start_time
        return parse_openclaw_to_run_result(stdout, stderr, proc.returncode, duration_s)
