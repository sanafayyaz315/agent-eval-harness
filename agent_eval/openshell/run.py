"""OpenShell backend orchestrator.

Usage: python -m agent_eval.openshell.run --config eval.yaml --model <model> --run-id <id>

Output written to: $AGENT_EVAL_RUNS_DIR/<eval-name>/<run-id>/
"""

import agent_eval._bootstrap  # noqa: F401 - required for entry points

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from agent_eval.agent.openclaw import build_openclaw_argv, parse_openclaw_to_case_dict
from agent_eval.config import EvalConfig
from agent_eval.events import (
    build_explicit_openclaw_session_key,
    events_from_openclaw_exec,
    parse_openclaw_session,
    parse_openclaw_trajectory_events,
    resolve_openclaw_session_file,
    resolve_openclaw_session_key_from_list,
)
from agent_eval.openshell.sandbox import OpenShellSandbox

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parents[2] / "skills" / "eval-run" / "scripts"

# Retained OpenClaw state under the sandbox workdir. ``agent exec`` deletes a
# temp state dir on exit unless ``--state-dir`` is set; without retained state
# Quay 2026.7.x has no harvestable trajectory (SQLite is wiped with the temp dir).
_OPENCLAW_STATE_DIR = Path("/sandbox/.openclaw")
_OPENCLAW_TMP_DIR = Path("/sandbox/tmp")
_FORGE_AI_GATEWAY_CA_PATH = Path("/sandbox/ca.crt")

# Graph tokens + mailbox identity. Also resolved from eval.yaml
# ``execution.env: $M365_*``; listed here so they reach sandbox exec even
# when a submission omits that block.
_M365_FORWARD_ENV = (
    "M365_ACCESS_TOKEN",
    "M365_USER",
    "M365_TENANT_ID",
    "M365_CLIENT_ID",
    "M365_CLIENT_SECRET",
)

# OpenClaw 8.1 redacts Bearer tokens in tool command text; Forge prompts use
# curl -H @$M365_AUTH_HEADER_FILE instead of expanding the token on argv.
_M365_HEADER_PATH = "/sandbox/tmp/m365.header"
_M365_GRAPH_CURL_PATH = "/sandbox/tmp/m365-graph-curl"
_M365_PLACEHOLDER_MARKERS = (
    "<replace-with",
    "changeme",
    "placeholder",
)


def _m365_usable(value: Optional[str]) -> bool:
    """True when an M365 env value is present and not a template placeholder."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    low = text.lower()
    return not any(marker in low for marker in _M365_PLACEHOLDER_MARKERS)


def _apply_scene_m365_user(config: EvalConfig) -> None:
    """Fill M365_USER from scene YAML when the orchestrator env omitted it."""
    scene = _load_scene(config)
    user = ((scene or {}).get("m365") or {}).get("user")
    if user and not _m365_usable(os.environ.get("M365_USER")):
        os.environ["M365_USER"] = str(user)


def _m365_required_keys(config: EvalConfig) -> List[str]:
    """Keys that must be set for live Graph (Forge) runs.

    Required when eval.yaml declares ``execution.env`` ``M365_*`` and/or the
    scene uses ``m365.seed: external``. Crabline/smolclaw scenes skip this.
    """
    needed: List[str] = []
    env = getattr(config.execution, "env", None) or {}
    if any(str(key).startswith("M365_") for key in env):
        needed.extend(["M365_ACCESS_TOKEN", "M365_USER"])
    scene = _load_scene(config)
    m365 = (scene or {}).get("m365") or {}
    if str(m365.get("seed") or "") == "external":
        for key in ("M365_ACCESS_TOKEN", "M365_USER"):
            if key not in needed:
                needed.append(key)
    return needed


def _ensure_m365_credentials(config: EvalConfig) -> None:
    """Fail before sandbox exec when Forge Graph credentials are missing.

    ``M365_AUTH_HEADER_FILE`` / ``M365_GRAPH_CURL`` are created inside the
    sandbox from ``M365_ACCESS_TOKEN``; they are not Secret keys.
    """
    _apply_scene_m365_user(config)
    needed = _m365_required_keys(config)
    if not needed:
        return
    saw_profile = os.environ.get("FORGE_SAW_PROFILE", "").strip()
    if saw_profile:
        logger.info(
            "M365 access delegated to SAW profile %s; no Graph token is required "
            "on the orchestrator",
            saw_profile,
        )
        return
    missing = [key for key in needed if not _m365_usable(os.environ.get(key))]
    if not missing:
        logger.info(
            "M365 Graph credentials present (%s)",
            ", ".join(needed),
        )
        return
    raise RuntimeError(
        "M365 Graph credentials are not available on the orchestrator "
        f"(missing or placeholder: {', '.join(missing)}). "
        "Forge scenes with m365.seed=external need a live token so the "
        "sandbox can install M365_AUTH_HEADER_FILE / M365_GRAPH_CURL. "
        "On cluster: create Secret openshell-credentials in the PipelineRun "
        "namespace with M365_ACCESS_TOKEN and M365_USER (optional: "
        "M365_TENANT_ID, M365_CLIENT_ID, M365_CLIENT_SECRET). "
        "Do not apply the placeholder template. Locally: export the same keys."
    )

# Environment variables to forward to sandbox (mirrors Harbor's _FORWARD_ENV)
_FORWARD_ENV = (
    # Provider config
    "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION",
    "GOOGLE_CLOUD_PROJECT", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK", "AWS_REGION",
    "OPENAI_BASE_URL", "OPENAI_MODEL",
    # API keys
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "OPENAI_API_KEY",
    # Live Microsoft Graph (Forge Outlook + calendar; not Crabline/smolclaw)
    *_M365_FORWARD_ENV,
)

# OpenClaw prefixes unqualified --model values as anthropic/<name> and then
# probes api.anthropic.com. Custom eval.yaml providers must win instead.
_ANTHROPIC_SANDBOX_ENV = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_VERTEX_PROJECT_ID",
)


def _resolve_provider_value(
    raw, *env_keys: str, preserve_unresolved_env: bool = False
) -> str:
    """Resolve a provider field: literal, $ENV, or first non-empty env fallback."""
    if isinstance(raw, str) and raw.startswith("$") and len(raw) > 1:
        resolved = os.environ.get(raw[1:], "")
        if resolved:
            return resolved
        if preserve_unresolved_env:
            return raw
        raw = ""
    if raw:
        return str(raw)
    for key in env_keys:
        found = os.environ.get(key)
        if found:
            return found
    return ""


def _openai_compat_base_url(url: str) -> str:
    """Ensure openai-completions baseUrl ends with /v1 (LiteLLM cluster URLs often omit it)."""
    url = (url or "").rstrip("/")
    if not url or url.endswith("/v1"):
        return url
    return url + "/v1"


def qualify_openclaw_model(model: str, providers: dict) -> str:
    """Return provider/id so OpenClaw does not default the model to anthropic/."""
    model = (model or "").strip()
    if not model or not providers:
        return model
    if "/" in model:
        return model
    for name, cfg in providers.items():
        ids = [
            m.get("id")
            for m in (cfg or {}).get("models") or []
            if isinstance(m, dict) and m.get("id")
        ]
        if model in ids:
            return f"{name}/{model}"
    first = next(iter(providers))
    return f"{first}/{model}"


def _openclaw_model_catalog_entry(model_id: str, name: str = "", api: str = "") -> dict:
    entry = {
        "id": model_id,
        "name": name or model_id,
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 200000,
        "maxTokens": 8192,
    }
    if api:
        entry["api"] = api
    return entry


def build_openclaw_eval_config(providers: dict, model: str) -> tuple:
    """Build /sandbox/openclaw-eval.json and the --model OpenClaw should receive.

    Pipeline --model is often a LiteLLM alias (claude-sonnet). OpenClaw needs
    that id listed under models.providers[<name>].models[] and a provider-
    qualified --model, otherwise it looks up anthropic/claude-sonnet.
    """
    qualified = qualify_openclaw_model(model, providers)
    requested_id = qualified.split("/", 1)[1] if "/" in qualified else qualified
    provider_name = (
        qualified.split("/", 1)[0] if "/" in qualified else next(iter(providers), "")
    )
    openclaw_config = {
        "agents": {
            "defaults": {
                "model": {"primary": qualified},
                "models": {qualified: {}},
            }
        },
        "models": {
            # Do not merge OpenClaw's built-in anthropic catalog (sandbox cannot
            # reach api.anthropic.com; catalog fetch times out and pollutes logs).
            "mode": "replace",
            "providers": {},
        },
    }
    for name, provider_cfg in providers.items():
        provider_cfg = provider_cfg or {}
        raw_base = provider_cfg.get("baseUrl", "")
        base_url = _resolve_provider_value(
            raw_base, "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"
        )
        if "inference.local" not in base_url:
            base_url = _openai_compat_base_url(base_url)
        api_key = (
            _resolve_provider_value(
                provider_cfg.get("apiKey", "empty"),
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                preserve_unresolved_env=True,
            )
            or "empty"
        )
        api = provider_cfg.get("api", "openai-completions")
        provider_entry = {
            "baseUrl": base_url,
            "apiKey": api_key,
            "api": api,
            "models": [],
        }
        # OpenClaw intentionally rejects private/special-use destinations
        # unless the provider explicitly opts in.  SAW's governed bridges are
        # exactly such destinations (host.containers.internal), so preserve
        # this schema-supported per-provider override when supplied by eval
        # configuration.  Do not make it a global default.
        request = provider_cfg.get("request")
        if isinstance(request, dict) and "allowPrivateNetwork" in request:
            provider_entry["request"] = {
                "allowPrivateNetwork": request["allowPrivateNetwork"],
            }
        seen = set()
        for m in provider_cfg.get("models") or []:
            if not isinstance(m, dict):
                continue
            mid = m.get("id", "")
            if not mid:
                continue
            seen.add(mid)
            entry = _openclaw_model_catalog_entry(
                mid, m.get("name", mid), m.get("api") or api
            )
            # Respect the declared model capabilities. Replacing reasoning
            # and output limits silently can exhaust the entire completion
            # budget before a reasoning model produces its final answer.
            for field in ("reasoning", "input", "cost", "contextWindow", "maxTokens"):
                if field in m:
                    entry[field] = m[field]
            provider_entry["models"].append(entry)
        if name == provider_name and requested_id and requested_id not in seen:
            provider_entry["models"].append(
                _openclaw_model_catalog_entry(requested_id, requested_id, api)
            )
        openclaw_config["models"]["providers"][name] = provider_entry
    return openclaw_config, qualified


async def _run_openclaw_llm_preflight(
    sandbox: OpenShellSandbox,
    sandbox_name: str,
    config_path: Path,
    qualified_model: str,
) -> None:
    """Make a minimal provider call, retrying one transient failure.

    This intentionally uses the same OpenClaw provider configuration that the
    case will use, but calls the OpenAI-compatible endpoint directly.  It
    separates model/network failures from agent tools, workspace, and memory
    failures and avoids spending the full case timeout on an unreachable LLM.
    """
    provider, model_id = qualified_model.split("/", 1)
    script = (
        "const fs=require('fs');"
        "const [configPath,providerName,modelId]=process.argv.slice(1);"
        "const c=JSON.parse(fs.readFileSync(configPath,'utf8'));"
        "const p=c.models.providers[providerName];"
        "if(!p||!p.baseUrl)throw new Error('provider config missing: '+providerName);"
        "const ref=typeof p.apiKey==='string'&&p.apiKey.match(/^\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}$/);"
        "const apiKey=ref?process.env[ref[1]]:p.apiKey;"
        "if(ref&&!apiKey)throw new Error('missing provider credential env: '+ref[1]);"
        "const url=p.baseUrl.replace(/\\/$/,'')+'/chat/completions';"
        "const headers={'content-type':'application/json'};"
        "if(apiKey&&apiKey!=='empty')headers.authorization='Bearer '+apiKey;"
        "const ctl=new AbortController();"
        "const timer=setTimeout(()=>ctl.abort(),30000);"
        "fetch(url,{method:'POST',headers,signal:ctl.signal,body:JSON.stringify({"
        "model:modelId,messages:[{role:'user',content:'How are you? Reply with exactly GLM_PREFLIGHT_OK.'}],"
        # GLM can spend a small completion budget on hidden reasoning before
        # producing visible text; 20 tokens can therefore yield an empty
        # content field even when the model is healthy.
        "max_tokens:512,temperature:0})})"
        ".then(async r=>{const text=await r.text();"
        "if(!r.ok)throw new Error('HTTP '+r.status+' '+text.slice(0,300));"
        "let body;try{body=JSON.parse(text)}catch{throw new Error('non-JSON response: '+text.slice(0,300))}"
        "const content=body.choices?.[0]?.message?.content||'';"
        "const summary={status:r.status,choices:Array.isArray(body.choices)?body.choices.length:0,"
        "usage:body.usage||null,error:body.error||null,contentPreview:content.slice(0,120)};"
        "if(!content.trim())throw new Error('empty model response '+JSON.stringify(summary));"
        "console.log('LLM_PREFLIGHT_OK provider='+providerName+' model='+modelId+' '+JSON.stringify(summary));"
        "}).finally(()=>clearTimeout(timer)).catch(e=>{console.error('LLM_PREFLIGHT_FAILED '+e.message);process.exitCode=1});"
    )
    logger.info(
        "Running in-sandbox LLM preflight provider=%s model=%s endpoint=<from config>",
        provider,
        model_id,
    )
    for attempt in range(2):
        result = await sandbox.exec(
            sandbox_name,
            ["node", "-e", script, str(config_path), provider, model_id],
            workdir="/sandbox",
            timeout_s=40,
        )
        output = ((result.stdout or "") + " " + (result.stderr or "")).strip()
        if not result.return_code:
            logger.info("In-sandbox LLM preflight passed: %s", output[:600])
            return
        transient = result.return_code == 124 or any(
            marker in output for marker in (
                "LLM_PREFLIGHT_FAILED This operation was aborted",
                "LLM_PREFLIGHT_FAILED HTTP 429 ",
                "LLM_PREFLIGHT_FAILED HTTP 502 ",
                "LLM_PREFLIGHT_FAILED HTTP 503 ",
                "LLM_PREFLIGHT_FAILED HTTP 504 ",
            )
        )
        if attempt == 0 and transient:
            logger.warning("Transient LLM preflight failure; retrying once for %s", qualified_model)
            await asyncio.sleep(2)
            continue
        raise RuntimeError(
            f"In-sandbox LLM preflight failed for {qualified_model}: {output[:600]}"
        )


def _child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a spawned python child.

    Each child is a fresh ``python3 script.py`` entry that has to activate
    ``.eval-venv`` itself. The bootstrap sentinel is designed to survive
    ``os.execv`` *within* one process; letting it cross into a child would make
    the child short-circuit activation and run without the venv's site-packages.
    """
    env = dict(os.environ)
    if extra:
        env.update(extra)
    # After the overrides, so `extra` cannot put the sentinel back.
    env.pop(agent_eval._bootstrap._SENTINEL, None)
    return env


async def _harvest_openclaw_events(
    sandbox: OpenShellSandbox,
    name: str,
    *,
    stdout_text: str,
    prompt: str,
    case_id: str,
    case_output: Path,
    sandbox_env: Dict[str, str],
) -> list:
    """Build AEH events from OpenClaw session JSONL, trajectory export, or envelope.

    Preference order:
    1. Legacy ``meta.agentMeta.sessionFile`` JSONL (pre-SQLite OpenClaw)
    2. ``openclaw sessions export-trajectory`` → ``events.jsonl`` (Quay 2026.7.x)
    3. Synthesize user/assistant from the compact ``agent exec`` envelope
    """
    openclaw_json: dict = {}
    try:
        openclaw_json = json.loads(stdout_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        start = (stdout_text or "").find("{")
        if start >= 0:
            try:
                openclaw_json = json.loads(stdout_text[start:])
            except (json.JSONDecodeError, ValueError):
                openclaw_json = {}

    # 1) Legacy session JSONL path
    session_file = resolve_openclaw_session_file(openclaw_json) if openclaw_json else None
    if session_file:
        try:
            cat_result = await sandbox.exec(name, ["cat", session_file])
            if cat_result.return_code == 0 and cat_result.stdout:
                events = parse_openclaw_session(cat_result.stdout)
                if events:
                    return events
        except Exception as e:
            logger.warning(f"Failed to read OpenClaw sessionFile for {case_id}: {e}")

    # 2) SQLite-era trajectory export (requires retained --state-dir)
    session_id = (openclaw_json or {}).get("sessionId") or ""
    if session_id:
        session_key = build_explicit_openclaw_session_key(session_id)
        export_name = f"aeh-{case_id}"
        try:
            export_result = await sandbox.exec(
                name,
                [
                    "openclaw",
                    "sessions",
                    "export-trajectory",
                    "--session-key",
                    session_key,
                    "--workspace",
                    "/sandbox",
                    "--output",
                    export_name,
                    "--json",
                ],
                workdir="/sandbox",
                env=sandbox_env,
                timeout_s=120,
            )
            if export_result.return_code != 0:
                # Fallback: resolve key via sessions list (match sessionId)
                list_result = await sandbox.exec(
                    name,
                    ["openclaw", "sessions", "--json"],
                    workdir="/sandbox",
                    env=sandbox_env,
                    timeout_s=60,
                )
                alt_key = resolve_openclaw_session_key_from_list(
                    list_result.stdout if list_result.return_code == 0 else "",
                    session_id,
                )
                if alt_key and alt_key != session_key:
                    session_key = alt_key
                    export_result = await sandbox.exec(
                        name,
                        [
                            "openclaw",
                            "sessions",
                            "export-trajectory",
                            "--session-key",
                            session_key,
                            "--workspace",
                            "/sandbox",
                            "--output",
                            export_name,
                            "--json",
                        ],
                        workdir="/sandbox",
                        env=sandbox_env,
                        timeout_s=120,
                    )

            if export_result.return_code == 0:
                summary = {}
                try:
                    summary = json.loads(export_result.stdout)
                except (json.JSONDecodeError, TypeError, ValueError):
                    start = (export_result.stdout or "").find("{")
                    if start >= 0:
                        try:
                            summary = json.loads(export_result.stdout[start:])
                        except (json.JSONDecodeError, ValueError):
                            summary = {}

                output_dir = summary.get("outputDir") or (
                    f"/sandbox/.openclaw/trajectory-exports/{export_name}"
                )
                events_path = f"{output_dir.rstrip('/')}/events.jsonl"
                cat_events = await sandbox.exec(name, ["cat", events_path])
                if cat_events.return_code == 0 and cat_events.stdout:
                    # Keep raw export for debugging / offline reparse
                    (case_output / "openclaw-trajectory-events.jsonl").write_text(
                        cat_events.stdout
                    )
                    events = parse_openclaw_trajectory_events(cat_events.stdout)
                    if events:
                        logger.info(
                            "Harvested %d events from OpenClaw trajectory for %s",
                            len(events),
                            case_id,
                        )
                        return events
                else:
                    logger.warning(
                        "Trajectory export for %s succeeded but events.jsonl "
                        "missing at %s (stderr=%s)",
                        case_id,
                        events_path,
                        (export_result.stderr or "")[:300],
                    )
            else:
                logger.warning(
                    "OpenClaw trajectory export failed for %s "
                    "(session_key=%s, code=%s): %s",
                    case_id,
                    session_key,
                    export_result.return_code,
                    (export_result.stderr or export_result.stdout or "")[:400],
                )
        except Exception as e:
            logger.warning(f"OpenClaw trajectory harvest failed for {case_id}: {e}")

    # 3) Compact envelope fallback (answer text only)
    return events_from_openclaw_exec(stdout_text, prompt=prompt)


def _sandbox_env(config: EvalConfig) -> Dict[str, str]:
    """Build environment dict to pass to sandbox exec.

    Forwards API keys, provider config, and merges execution.env + runner.env.
    """
    env = {}
    # Forward allowlisted env vars (API keys, provider config)
    for key in _FORWARD_ENV:
        value = os.environ.get(key)
        if value:
            env[key] = value
    # Merge execution.env and runner.env (runner wins on collision)
    if config.execution.env:
        for key, value in config.execution.env.items():
            if value is not None:
                # Resolve $VAR references
                if isinstance(value, str) and value.startswith("$"):
                    resolved = os.environ.get(value[1:])
                    if resolved:
                        env[key] = resolved
                else:
                    env[key] = str(value)
    if config.runner.env:
        for key, value in config.runner.env.items():
            if value is not None:
                if isinstance(value, str) and value.startswith("$"):
                    resolved = os.environ.get(value[1:])
                    if resolved:
                        env[key] = resolved
                else:
                    env[key] = str(value)
    return env


def _safe_endpoint(value: str) -> str:
    """Return endpoint metadata without query strings or credentials."""
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.split("?", 1)[0]
    return f"{parsed.scheme}://{parsed.hostname or parsed.netloc}:{parsed.port or ''}{parsed.path}"


def _log_model_diagnostics(
    case_id: str, model: str, sandbox_env: Dict[str, str], sandbox_name: str
) -> None:
    """Log model routing metadata while never logging credential values."""
    endpoints = {
        key: _safe_endpoint(value)
        for key, value in sandbox_env.items()
        if key.endswith("BASE_URL") and value
    }
    present = sorted(
        key for key, value in sandbox_env.items()
        if value and (key.endswith("API_KEY") or key.endswith("TOKEN") or "BEARER" in key)
    )
    proxy = next(
        (sandbox_env.get(key) for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy") if sandbox_env.get(key)),
        "<none>",
    )
    logger.info(
        "Model routing diagnostics case=%s sandbox=%s model=%s endpoints=%s proxy=%s credential_vars=%s",
        case_id,
        sandbox_name,
        model,
        endpoints or {},
        _safe_endpoint(proxy),
        present,
    )


async def _stage_forge_ai_gateway_ca(
    sandbox: OpenShellSandbox, name: str, sandbox_env: Dict[str, str]
) -> None:
    """Stage the optional Forge AI bridge CA into a sandbox for Node.

    The CI orchestrator mounts only the public CA from its namespace.  The
    gateway-issued provider bearer remains inside the sandbox; this helper
    merely lets Node validate the bridge's private TLS chain.
    """
    configured = os.environ.get("AGENT_EVAL_FORGE_AI_GATEWAY_CA_FILE", "").strip()
    if not configured:
        return
    source = Path(configured)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(
            "AGENT_EVAL_FORGE_AI_GATEWAY_CA_FILE does not name a readable CA file: "
            f"{source}"
        )
    parent = str(_FORGE_AI_GATEWAY_CA_PATH.parent)
    mkdir = await sandbox.exec(name, ["mkdir", "-p", parent])
    if mkdir.return_code:
        raise RuntimeError(f"Could not create Forge CA directory in sandbox {name}")
    # Use the image's Node runtime to write the CA into the sandbox workspace.
    # OpenShell's upload helper can report success while placing the file in a
    # path that is not visible to the subsequent Node process.
    writer = await sandbox.exec(
        name,
        [
            "node",
            "-e",
            "require('fs').writeFileSync(process.argv[1], require('fs').readFileSync(0))",
            str(_FORGE_AI_GATEWAY_CA_PATH),
        ],
        stdin=source.read_bytes(),
    )
    if writer.return_code:
        raise RuntimeError(f"Could not stage Forge AI gateway CA in sandbox {name}")
    sandbox_env["NODE_EXTRA_CA_CERTS"] = str(_FORGE_AI_GATEWAY_CA_PATH)
    logger.info("Forge AI gateway CA staged for Node TLS validation")


async def _install_m365_file_auth(
    sandbox: OpenShellSandbox,
    name: str,
    sandbox_env: Dict[str, str],
) -> None:
    """Install Graph header-file auth inside the sandbox (OpenClaw 8.1-safe).

    Forge prompts tell the agent to call Graph with
    ``curl -H @$M365_AUTH_HEADER_FILE`` (or ``$M365_GRAPH_CURL``). Expanding
    ``M365_ACCESS_TOKEN`` on the tool command line is redacted to ``***``.
    Tokens themselves are still passed in ``sandbox_env`` for MCP/CLI tools.
    """
    token = sandbox_env.get("M365_ACCESS_TOKEN")
    if not token:
        return

    mkdir = await sandbox.exec(
        name, ["mkdir", "-p", str(_OPENCLAW_TMP_DIR)],
    )
    if mkdir.return_code:
        logger.warning(
            "Could not create %s for M365 auth files (rc=%s)",
            _OPENCLAW_TMP_DIR,
            mkdir.return_code,
        )
        return

    header = f"Authorization: Bearer {token}\n"
    written = await sandbox.exec(
        name,
        ["tee", _M365_HEADER_PATH],
        stdin=header.encode(),
    )
    if written.return_code:
        logger.warning(
            "Could not write M365 auth header in sandbox (rc=%s)",
            written.return_code,
        )
        return

    sandbox_env["M365_AUTH_HEADER_FILE"] = _M365_HEADER_PATH

    wrapper = (
        "#!/bin/sh\n"
        f'exec curl -sS -H @"{_M365_HEADER_PATH}" "$@"\n'
    )
    curl_written = await sandbox.exec(
        name,
        ["tee", _M365_GRAPH_CURL_PATH],
        stdin=wrapper.encode(),
    )
    if curl_written.return_code == 0:
        await sandbox.exec(name, ["chmod", "+x", _M365_GRAPH_CURL_PATH])
        sandbox_env["M365_GRAPH_CURL"] = _M365_GRAPH_CURL_PATH

    present = [k for k in _M365_FORWARD_ENV if sandbox_env.get(k)]
    logger.info(
        "M365 file-auth installed in sandbox (%s, header=%s)",
        ", ".join(present),
        _M365_HEADER_PATH,
    )


def _resolve_prompt(config: EvalConfig, case_data: dict) -> str:
    """Resolve prompt template using Jinja2 or str.format().

    Mirrors execute.py's _resolve_arguments for template parity.
    """
    template = config.execution.prompt or config.execution.arguments or ""
    if not template:
        return case_data.get("prompt", "")

    # Auto-detect Jinja2 syntax
    if "{{" in template or "{%" in template:
        try:
            from jinja2 import StrictUndefined, Template, UndefinedError
        except ImportError:
            raise ImportError(
                "Jinja2 is required for {{ }} template syntax. "
                "Install with: pip install jinja2"
            )
        try:
            jinja_template = Template(template, undefined=StrictUndefined)
            return jinja_template.render(input=case_data).strip()
        except UndefinedError as e:
            raise ValueError(f"Undefined variable in template: {e}")
    else:
        # Simple str.format() with {field} placeholders
        import re
        def replacer(match):
            field = match.group(1)
            optional = field.endswith("?")
            if optional:
                field = field[:-1]
            if field in case_data:
                return str(case_data[field])
            elif optional:
                return ""
            else:
                raise ValueError(f"Missing required field: {field}")
        return re.sub(r"\{(\w+\??)\}", replacer, template).strip()


def _load_scene(config: EvalConfig) -> Optional[dict]:
    """Load scene YAML if configured. Return parsed dict or None."""
    raw_path = config.config_path
    if not raw_path:
        return None
    with open(raw_path) as f:
        raw = yaml.safe_load(f) or {}
    scene_name = raw.get("scene")
    if not scene_name:
        return None
    scene_path = raw_path.parent / "scenes" / f"{scene_name}.yaml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")
    logger.info("Loading scene: %s", scene_path)
    return yaml.safe_load(scene_path.read_text(encoding="utf-8")) or {}


def _setup_scene(config: EvalConfig, output_dir: Path) -> bool:
    """Apply scene YAML once at run start.

    Crabline seeds Slack-mock messages; smolclaw seeds Gmail/Calendar mocks.
    Forge M365 scenes leave those lists empty (``seed: external``): the live
    Graph mailbox is already populated and this function only records that
    fact. Graph tokens are forwarded later via ``_sandbox_env``.
    """
    scene = _load_scene(config)
    if not scene:
        return False

    from agent_eval.openshell.crabline_seed import seed_crabline_for_scene
    from agent_eval.openshell.smolclaw_seed import seed_smolclaw_for_scene

    all_meta = {}

    crabline_seeds = scene.get("crabline_seeds") or []
    if crabline_seeds:
        logger.info("Seeding %d Crabline messages for scene...", len(crabline_seeds))
        crabline_meta = seed_crabline_for_scene(crabline_seeds)
        all_meta["crabline"] = crabline_meta

    smolclaw_seeds = scene.get("smolclaw_seeds") or []
    if smolclaw_seeds:
        logger.info("Seeding %d smolclaw items for scene...", len(smolclaw_seeds))
        smolclaw_meta = seed_smolclaw_for_scene(smolclaw_seeds)
        all_meta["smolclaw"] = smolclaw_meta

    m365 = scene.get("m365") or {}
    slack = scene.get("slack") or {}
    if m365:
        seed_mode = str(m365.get("seed") or "")
        # SAW-delegated runs intentionally do not expose a Graph bearer token
        # to the orchestrator; the sandbox receives governed access through
        # the selected SAW profile instead.
        saw_profile = os.environ.get("FORGE_SAW_PROFILE", "").strip()
        token_present = bool(os.environ.get("M365_ACCESS_TOKEN")) or bool(saw_profile)
        all_meta["m365"] = {
            "user": m365.get("user"),
            "seed": seed_mode,
            "access_token": (
                "delegated" if saw_profile and not os.environ.get("M365_ACCESS_TOKEN")
                else "present" if token_present else "missing"
            ),
            "slack_enabled": bool(slack.get("enabled")),
        }
        if seed_mode == "external":
            logger.info(
                "M365 seed=external: mailbox is pre-seeded (user=%s); "
                "not seeding Crabline/smolclaw — forwarding Graph tokens only",
                m365.get("user") or "unset",
            )
            if not token_present:
                # _ensure_m365_credentials raises before cases run; keep the
                # metadata file so operators can see access_token=missing.
                logger.warning(
                    "M365 seed=external but M365_ACCESS_TOKEN is not set "
                    "on the orchestrator"
                )
        else:
            logger.info(
                "M365 scene seed=%s user=%s",
                seed_mode or "unset",
                m365.get("user") or "unset",
            )

    # Persist scene seed metadata for debugging (never write token values)
    (output_dir / "scene-seed.json").write_text(
        json.dumps(all_meta, indent=2, default=str), encoding="utf-8"
    )
    slack_state = "enabled" if slack.get("enabled") else "disabled"
    m365_state = "none"
    if m365:
        m365_state = str(m365.get("seed") or "configured")
        if m365.get("user"):
            m365_state = f"{m365_state} ({m365['user']})"
    logger.info(
        "Scene ready: slack=%s crabline=%d smolclaw=%d m365=%s",
        slack_state,
        len(crabline_seeds),
        len(smolclaw_seeds),
        m365_state,
    )
    return True


async def run_openshell(
    config_path: Path,
    model: str,
    run_id: str,
    parallelism: int = 1,
    keep_sandbox: bool = False,
    cases: Optional[List[str]] = None,
    no_llm_judges: bool = False,
) -> int:
    """Execute evaluation in OpenShell sandboxes.

    Pipeline: workspace.py -> sandbox lifecycle -> collect -> score -> report -> regression

    Args:
        config_path: Path to eval.yaml.
        model: Model identifier.
        run_id: Run identifier.
        parallelism: Number of concurrent cases.
        keep_sandbox: Keep sandboxes after trial for debugging.
        cases: Optional list of case IDs to run (default: all).
        no_llm_judges: Skip LLM judges.

    Returns:
        Exit code (non-zero on regression).
    """
    config = EvalConfig.from_yaml(config_path)
    sandbox_mgr = OpenShellSandbox.from_env()

    # Resolve to absolute path once - subprocesses run with different cwd
    abs_config_path = config_path.resolve()

    # Find project root (repository root) for consistent path resolution.
    # Walk up from config file until we find a marker (e.g., .git, pyproject.toml).
    project_root = abs_config_path.parent
    for parent in [abs_config_path.parent] + list(abs_config_path.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            project_root = parent
            break

    image = os.environ.get("AGENT_EVAL_OPENSHELL_IMAGE")
    if not image:
        raise RuntimeError(
            "AGENT_EVAL_OPENSHELL_IMAGE environment variable is required"
        )

    runs_dir = (
        Path(os.environ.get("AGENT_EVAL_RUNS_DIR", "eval/runs")) / config.eval_name()
    ).resolve()  # Use absolute path to avoid cwd issues
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = runs_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage workspaces (subprocess workspace.py, parse WORKSPACE line from stdout)
    workspace_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "workspace.py"),
        "--config",
        str(abs_config_path),
        "--run-id",
        run_id,
    ]
    if cases:
        workspace_cmd.extend(["--cases"] + cases)

    logger.info(f"Staging workspaces: {' '.join(workspace_cmd)}")
    result = subprocess.run(
        workspace_cmd,
        capture_output=True,
        text=True,
        check=True,
        env=_child_env(),
        cwd=project_root,
    )

    workspace_root = None
    for line in result.stdout.splitlines():
        if line.startswith("WORKSPACE:"):
            workspace_root = Path(line.split(":", 1)[1].strip())
            break
    if not workspace_root or not workspace_root.exists():
        raise RuntimeError(
            f"workspace.py did not emit valid WORKSPACE path: {result.stdout}"
        )

    case_dirs = sorted((workspace_root / "cases").iterdir())
    start_time = time.monotonic()

    # Scene seeding — seed once before all cases. Fail closed when the
    # submission needs live Graph but M365_* is unset (otherwise OpenClaw
    # refuses and judges score 1/5 with an empty briefing).
    _ensure_m365_credentials(config)
    scene_active = _setup_scene(config, output_dir)

    # 2. Run cases in sandboxes (parallel, with error isolation)
    sem = asyncio.Semaphore(parallelism)
    tasks = [
        _run_case(
            sandbox_mgr, config, case_dir, model, image, output_dir, sem, keep_sandbox, scene_active
        )
        for case_dir in case_dirs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_case = {}
    n_failed = 0
    for case_dir, case_result in zip(case_dirs, results):
        case_id = case_dir.name
        if isinstance(case_result, Exception):
            logger.error(f"Case {case_id} failed: {case_result}")
            per_case[case_id] = {"exit_code": 1, "error": str(case_result)}
            n_failed += 1
        else:
            per_case[case_id] = case_result
            # Count non-zero exits (including timeout 124) as failures
            if case_result.get("exit_code", 0) != 0:
                n_failed += 1

    wall_clock_s = time.monotonic() - start_time

    # 3. Write suite-level run_result.json
    # Aggregate cost and tokens from per_case results
    total_cost = sum(
        c.get("cost_usd", 0) or 0 for c in per_case.values() if isinstance(c, dict)
    )
    total_input = sum(
        c.get("token_usage", {}).get("input", 0) or 0
        for c in per_case.values() if isinstance(c, dict)
    )
    total_output = sum(
        c.get("token_usage", {}).get("output", 0) or 0
        for c in per_case.values() if isinstance(c, dict)
    )

    suite_result = {
        "execution_mode": "openshell",
        "agent": "openshell:openclaw",
        "model": model,
        "exit_code": 0 if n_failed == 0 else 1,
        "n_cases": len(case_dirs),
        "n_failed": n_failed,
        "per_case": per_case,
        "wall_clock_s": round(wall_clock_s, 1),
        "cost_usd": round(total_cost, 4) if total_cost else None,
        "token_usage": {"input": total_input, "output": total_output},
    }
    with open(output_dir / "run_result.json", "w") as f:
        json.dump(suite_result, f, indent=2)

    # 4. Collect outputs (subprocess)
    logger.info("Collecting outputs...")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "collect.py"),
            "--config",
            str(abs_config_path),
            "--workspace",
            str(workspace_root),
            "--output",
            str(output_dir),
        ],
        check=True,
        env=_child_env(),
        cwd=project_root,
    )

    # 5. Score (subprocess judges subcommand)
    logger.info("Running judges...")
    score_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "score.py"),
        "judges",
        "--run-id",
        run_id,
        "--config",
        str(abs_config_path),
        "--workspace",
        str(workspace_root),
        "--model",
        model,
    ]
    if no_llm_judges:
        score_cmd.append("--no-llm-judges")
    score_result = subprocess.run(score_cmd, env=_child_env(), cwd=project_root)
    score_exit_code = score_result.returncode

    # 6. Generate report (subprocess) - always generate even if scoring had issues
    logger.info("Generating report...")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "report.py"),
            "--run-id",
            run_id,
            "--config",
            str(abs_config_path),
        ],
        check=True,
        env=_child_env(),
        cwd=project_root,
    )

    # 7. Regression detection (in-process, like Harbor/EvalHub)
    if config.thresholds:
        summary_path = output_dir / "summary.yaml"
        if summary_path.exists():
            summary = yaml.safe_load(summary_path.read_text())
            spec = importlib.util.spec_from_file_location(
                "score", SCRIPTS_DIR / "score.py"
            )
            score_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(score_mod)

            regressions = score_mod.detect_regressions(
                summary.get("judges", {}), config.thresholds
            )
            if regressions:
                for r in regressions:
                    logger.warning(
                        f"REGRESSION [{r.judge_name}] {r.metric}: "
                        f"{r.baseline_value} -> {r.current_value}"
                    )
                return 1

    report_path = output_dir / "report.html"
    if report_path.exists():
        logger.info(f"Report: {report_path}")
    logger.info(f"Run complete: {output_dir}")

    # Exit non-zero if any cases failed (exception or non-zero exit code)
    if n_failed > 0:
        logger.warning(f"{n_failed}/{len(case_dirs)} cases failed")
        return 1
    
    # Propagate scoring exit code (e.g., regression detection in score.py)
    if score_exit_code != 0:
        return score_exit_code

    return 0


async def _openclaw_output_present(sandbox: OpenShellSandbox, name: str) -> bool:
    """Skip only a confirmed absent optional output; retain other failures."""
    probe = await sandbox.exec(name, [
        "node", "-e",
        "try { require('node:fs').statSync('/sandbox/output'); } "
        "catch (e) { process.exit(e.code === 'ENOENT' ? 3 : 2); }",
    ])
    if probe.return_code == 3:
        logger.info("No optional /sandbox/output in %s; collecting OpenClaw response from stdout", name)
        return False
    # Permission/probe errors must still reach download's normal diagnostics.
    return True


async def _run_case(
    sandbox: OpenShellSandbox,
    config: EvalConfig,
    staged_case: Path,
    model: str,
    image: str,
    output_dir: Path,
    sem: asyncio.Semaphore,
    keep: bool,
    scene_active: bool = False,
) -> dict:
    """Run single case in sandbox.

    - Outputs (config.outputs[].path) -> staged_case (workspace) for collect.py
    - Logs (stdout.log, stderr.log, run_result.json) -> output_dir/cases/<id>/ directly

    Args:
        sandbox: OpenShellSandbox instance.
        config: EvalConfig instance.
        staged_case: Path to staged case directory (workspace_root/cases/<id>).
        model: Model identifier.
        image: Container image with OpenClaw.
        output_dir: Run output directory (runs/<run-id>/).
        sem: Semaphore for parallelism control.
        keep: Keep sandbox after trial.
        scene_active: Skip per-case seeding when True (scene was seeded at run start).

    Returns:
        Case result dict for suite per_case.
    """
    case_id = staged_case.name
    case_output = output_dir / "cases" / case_id
    case_output.mkdir(parents=True, exist_ok=True)

    async with sem:
        # OpenShell sandbox names max 19 chars: prefix(2) + hex(8) + dash + digits
        name = f"e-{uuid.uuid4().hex[:8]}-{case_id[-3:]}"
        start_time = time.monotonic()
        try:
            logger.info(f"Creating sandbox {name} for case {case_id}")
            await sandbox.create(name, image)
            forge_image = os.environ.get("AGENT_EVAL_OPENSHELL_WORKSPACE") == "forge-image"
            if forge_image:
                from agent_eval.openshell.forge import prepare_forge_sandbox

                ca_file = os.environ.get("AGENT_EVAL_FORGE_AI_GATEWAY_CA_FILE", "")
                if not ca_file:
                    raise ValueError("Forge image workspace requires AGENT_EVAL_FORGE_AI_GATEWAY_CA_FILE")
                user_file = os.environ.get("AGENT_EVAL_FORGE_USER_FILE", "").strip()
                await prepare_forge_sandbox(
                    sandbox, name, Path(ca_file),
                    user_file=Path(user_file) if user_file else None,
                )

            # Upload files individually. OpenShell nests directory uploads at
            # the destination (for example, uploading ``skills`` to
            # /sandbox/skills can produce /sandbox/skills/skills), so
            # directory-level uploads break OpenClaw's literal paths.
            for entry in sorted(
                (path for path in staged_case.rglob("*")
                 if path.is_file() and ".git" not in path.relative_to(staged_case).parts),
                key=lambda path: str(path.relative_to(staged_case)),
            ):
                relative = entry.relative_to(staged_case)
                remote_path = f"/sandbox/{relative}"
                # The published OpenClaw image may already provide runtime
                # files (for example /sandbox/bin/m365). OpenShell upload
                # cannot replace a path whose parent is a file/directory, so
                # preserve an image-provided path and only stage missing files.
                exists = await sandbox.exec(
                    name,
                    ["sh", "-c", f"test -e {shlex.quote(remote_path)}"],
                )
                if exists.return_code == 0:
                    logger.info("Preserving image-provided workspace path %s", remote_path)
                    continue
                # CLI upload's destination is a directory, not a filename.
                await sandbox.upload(name, entry, str(Path(remote_path).parent))


            input_yaml_path = staged_case / "input.yaml"
            if input_yaml_path.exists():
                input_yaml = yaml.safe_load(input_yaml_path.read_text()) or {}
            else:
                input_yaml = {}

            # Resolve prompt template (Jinja2 or str.format)
            prompt = _resolve_prompt(config, input_yaml)
            # Older Forge prompts used the host-side placeholder literally.
            # OpenClaw does not expand it in read-tool arguments; normalize it
            # at the harness boundary while retaining support for those cases.
            prompt = prompt.replace("$WORKSPACE_DIR", "/sandbox")
            prompt = prompt.replace("${WORKSPACE_DIR}", "/sandbox")
            system_prompt = getattr(config.runner, "system_prompt", None)
            if system_prompt and str(system_prompt).strip():
                # OpenClaw agent exec has no --append-system-prompt; prepend.
                prompt = f"{str(system_prompt).strip()}\n\n{prompt}"

            # Skip per-case seeding when a scene was seeded at run start
            sandbox_env_extra: dict[str, str] = {}
            if not scene_active:
                # Optional host-side seeds (Crabline Slack / smolclaw Gmail|Calendar)
                from agent_eval.openshell.crabline_seed import (
                    load_case_annotations,
                    seed_crabline_for_case,
                )
                from agent_eval.openshell.smolclaw_seed import seed_smolclaw_for_case

                case_annotations = load_case_annotations(config, case_id)
                try:
                    seed_meta = seed_crabline_for_case(case_annotations)
                except Exception as e:
                    logger.error("Crabline seed failed for %s: %s", case_id, e)
                    raise
                if seed_meta:
                    seed_path = case_output / "crabline-seed.json"
                    seed_path.write_text(json.dumps(seed_meta, indent=2), encoding="utf-8")
                    # Non-secret metadata for the agent (not the seeded text/code).
                    sandbox_env_extra.update(
                        {
                            "CRABLINE_SEED_CHANNEL": str(seed_meta.get("channel") or ""),
                            "CRABLINE_SEED_TS": str(seed_meta.get("ts") or ""),
                            "CRABLINE_SEED_OLDEST": str(seed_meta.get("oldest_ts") or ""),
                            "CRABLINE_CASE_USER": str(
                                case_annotations.get("slack_user")
                                or (case_annotations.get("crabline_seed") or {}).get("users")
                                or ""
                            ),
                        }
                    )
                else:
                    slack_user = str(case_annotations.get("slack_user") or "")
                    if slack_user:
                        sandbox_env_extra["CRABLINE_CASE_USER"] = slack_user

                try:
                    smol_meta = seed_smolclaw_for_case(case_annotations)
                except Exception as e:
                    logger.error("smolclaw seed failed for %s: %s", case_id, e)
                    raise
                if smol_meta:
                    (case_output / "smolclaw-seed.json").write_text(
                        json.dumps(smol_meta, indent=2), encoding="utf-8"
                    )
                    kind = str(smol_meta.get("kind") or "")
                    if kind == "calendar" and smol_meta.get("event_id"):
                        sandbox_env_extra["SMOLCLAW_SEED_EVENT_ID"] = str(
                            smol_meta["event_id"]
                        )
                    if kind == "gmail" and smol_meta.get("message_id"):
                        sandbox_env_extra["SMOLCLAW_SEED_MESSAGE_ID"] = str(
                            smol_meta["message_id"]
                        )
                        if smol_meta.get("thread_id"):
                            sandbox_env_extra["SMOLCLAW_SEED_THREAD_ID"] = str(
                                smol_meta["thread_id"]
                            )

            # Build env to forward to sandbox (API keys + config env + M365_*)
            sandbox_env = _sandbox_env(config)
            sandbox_env.update({k: v for k, v in sandbox_env_extra.items() if v})
            if forge_image:
                # This profile uses supervisor-injected provider placeholders,
                # not raw orchestrator credentials or AEH-created tool wrappers.
                for key in ("OPENAI_API_KEY", "M365_ACCESS_TOKEN", "M365_CLIENT_SECRET"):
                    sandbox_env.pop(key, None)
            else:
                await _stage_forge_ai_gateway_ca(sandbox, name, sandbox_env)
                await _install_m365_file_auth(sandbox, name, sandbox_env)

            # Build command based on runner type
            openclaw_model = model
            # Default depends on whether providers are configured (OpenClaw) or not (Claude Code)
            if hasattr(config.runner, 'type') and config.runner.type:
                runner_type = config.runner.type
            elif getattr(config.runner, 'providers', None):
                runner_type = "openclaw"  # Providers configured = OpenClaw
            else:
                runner_type = "claude-code"  # Default for simple cases
            if runner_type == "cli":
                # CLI runner: use command from config with {args}/{case_id}
                # substitution. Use /bin/sh (not bash) — Quay OpenClaw and
                # many minimal images do not ship bash (exit 127).
                # Case files upload to /sandbox/<case_id>/ (OpenShell nests dirs).
                cli_command = config.runner.command
                if cli_command:
                    if isinstance(cli_command, list):
                        cli_command = " ".join(cli_command)
                    cli_command = (
                        cli_command.replace("{args}", prompt)
                        .replace("{case_id}", case_id)
                    )
                    cmd = ["/bin/sh", "-c", cli_command]
                    stdin_data = None
                else:
                    raise ValueError("CLI runner requires 'command' in runner config")
            elif runner_type == "claude-code":
                # Claude Code runner
                cmd = [
                    "claude",
                    "--print",
                    "--output-format", "stream-json",
                    "--model", model,
                    "--max-turns", "1",  # Single turn for simple prompts
                ]
                # Add dangerously-skip-permissions for headless execution
                cmd.append("--dangerously-skip-permissions")
                stdin_data = prompt.encode()
            else:
                # OpenClaw runner (default)
                # Custom providers (e.g. inference.local) must be registered in
                # openclaw.json and passed via --config. --auth-env-only skips
                # config entirely (OpenClaw docs), so it cannot be used together
                # with models.providers — that is why Quay beta.7 reported
                # "Unknown model" for inference/claude-sonnet-4.
                providers = getattr(config.runner, 'providers', None)
                config_path = None
                auth_env_only = True
                openclaw_model = model
                await sandbox.exec(
                    name,
                    [
                        "mkdir",
                        "-p",
                        str(_OPENCLAW_STATE_DIR),
                        str(_OPENCLAW_TMP_DIR),
                    ],
                )
                sandbox_env["HOME"] = "/sandbox"
                sandbox_env["OPENCLAW_STATE_DIR"] = str(_OPENCLAW_STATE_DIR)
                sandbox_env["TMPDIR"] = str(_OPENCLAW_TMP_DIR)
                if providers:
                    openclaw_config, openclaw_model = build_openclaw_eval_config(
                        providers, model
                    )
                    # Custom providers are openai-compatible (LiteLLM / inference.local).
                    # Anthropic env makes OpenClaw discover api.anthropic.com.
                    for key in _ANTHROPIC_SANDBOX_ENV:
                        sandbox_env.pop(key, None)
                    config_path = Path("/sandbox/openclaw-eval.json")
                    config_json = json.dumps(openclaw_config)
                    provider_names = list(
                        openclaw_config["models"]["providers"].keys()
                    )
                    logger.info(
                        "OpenClaw eval config path=%s model=%s providers=%s",
                        config_path,
                        openclaw_model,
                        provider_names,
                    )
                    # Quay OpenClaw image has node but not python3
                    await sandbox.exec(
                        name,
                        ["tee", str(config_path)],
                        stdin=config_json.encode(),
                    )
                    # SAW providers inject their bearer only into the sandbox.
                    # Preserve a native environment reference in the readable
                    # config; never materialize the injected credential there.
                    await sandbox.exec(
                        name,
                        [
                            "node",
                            "-e",
                            "const fs=require('fs');"
                            "const p='/sandbox/openclaw-eval.json';"
                            "const c=JSON.parse(fs.readFileSync(p,'utf8'));"
                            "for(const v of Object.values(c.models.providers)){"
                            "if(typeof v.apiKey==='string'&&/^\\$[A-Za-z_][A-Za-z0-9_]*$/.test(v.apiKey)){"
                            "const key=v.apiKey.slice(1),value=process.env[key];"
                            "if(!value)throw new Error('missing sandbox provider credential: '+key);"
                            "v.apiKey='${'+key+'}';}}"
                            "fs.writeFileSync(p,JSON.stringify(c));",
                        ],
                    )
                    await _run_openclaw_llm_preflight(
                        sandbox, name, config_path, openclaw_model
                    )
                    sandbox_env["OPENCLAW_CONFIG_PATH"] = str(config_path)
                    auth_env_only = False

                effort = config.runner.effort
                if not effort and config.runner.settings:
                    effort = config.runner.settings.get("effort")

                # Pass --state-dir so agent exec keeps SQLite (default temp state
                # is deleted on exit). Same path as OPENCLAW_STATE_DIR under
                # /sandbox (Landlock read_write). Needed for trajectory export.
                cmd = build_openclaw_argv(
                    model=openclaw_model,
                    timeout_s=config.execution.timeout,
                    effort=effort,
                    cwd=Path("/sandbox"),
                    auth_env_only=auth_env_only,
                    config_path=config_path,
                    state_dir=_OPENCLAW_STATE_DIR,
                )
                # Prompt is positional argument in 'agent exec' format
                cmd.append(prompt)
                stdin_data = None

            logger.info(
                "Executing case %s in sandbox %s argv=%s",
                case_id,
                name,
                cmd[:-1] if len(cmd) > 1 else cmd,
            )
            timeout = (config.execution.timeout or 600) + 60
            result = await sandbox.exec(
                name,
                cmd,
                workdir="/sandbox",
                stdin=stdin_data if runner_type != "cli" else None,
                env=sandbox_env,
                timeout_s=timeout,
            )
            _log_model_diagnostics(case_id, openclaw_model, sandbox_env, name)
            duration_s = time.monotonic() - start_time
            if result.return_code:
                logger.warning(
                    "Case %s sandbox exec rc=%s duration=%.2fs stderr=%r stdout=%r",
                    case_id,
                    result.return_code,
                    duration_s,
                    (result.stderr or "")[:800],
                    (result.stdout or "")[:400],
                )

            for output in config.outputs or []:
                if output.path:
                    # OpenClaw's response is collected from stdout below. Its
                    # conventional output directory is optional, unlike other
                    # explicitly requested artifacts.
                    if runner_type == "openclaw" and output.path == "output":
                        if not await _openclaw_output_present(sandbox, name):
                            continue
                    try:
                        await sandbox.download(
                            name, f"/sandbox/{output.path}", staged_case / output.path
                        )
                    except Exception as e:
                        # OpenClaw prompt cases often never create /sandbox/output;
                        # AEH writes response.txt from the exec envelope on the host.
                        err = str(e)
                        if (runner_type == "openclaw" and output.path == "output"
                                and "No such file or directory" in err):
                            logger.info(
                                "No sandbox %s to download for %s (ok for openclaw)",
                                output.path,
                                case_id,
                            )
                        else:
                            logger.warning(f"Failed to download {output.path}: {e}")

            # Parse output based on runner type
            if runner_type == "openclaw":
                case_result = parse_openclaw_to_case_dict(
                    result.stdout.encode(),
                    result.stderr.encode(),
                    result.return_code,
                    duration_s,
                )
                # Extract response text and write to output/ directory (following existing convention)
                response_text = case_result.get("response_text", "")
                output_dir = staged_case / "output"
                output_dir.mkdir(exist_ok=True)
                (output_dir / "response.txt").write_text(response_text)

                try:
                    events = await _harvest_openclaw_events(
                        sandbox,
                        name,
                        stdout_text=result.stdout,
                        prompt=prompt,
                        case_id=case_id,
                        case_output=case_output,
                        sandbox_env=sandbox_env,
                    )
                    with open(case_output / "events.json", "w") as f:
                        json.dump(events, f, indent=2)
                    logger.debug(
                        "Generated events.json for %s with %d events",
                        case_id,
                        len(events),
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate events.json for {case_id}: {e}")
            else:
                # Generic result for cli/claude-code runners
                case_result = {
                    "exit_code": result.return_code,
                    "duration_s": round(duration_s, 1),
                    "token_usage": {"input": 0, "output": 0},
                    "cost_usd": None,
                    "num_turns": 1,
                    "response_text": result.stdout,
                    "stderr": result.stderr,
                }
                # Ensure judges/collect see a response even if sandbox
                # did not create outputs.path (e.g. CLI stdout-only).
                host_output = staged_case / "output"
                host_output.mkdir(exist_ok=True)
                response_file = host_output / "response.txt"
                if not response_file.exists() or not response_file.read_text().strip():
                    response_file.write_text(result.stdout or "")

            with open(case_output / "run_result.json", "w") as f:
                json.dump(case_result, f, indent=2)
            (case_output / "stdout.log").write_text(result.stdout)
            (case_output / "stderr.log").write_text(result.stderr)

            logger.info(f"Case {case_id} completed with exit code {result.return_code}")
            return case_result

        finally:
            if not keep:
                await sandbox.delete(name)
            else:
                logger.info(f"Kept sandbox {name}: openshell sandbox connect {name}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenShell backend for agent-eval-harness"
    )
    parser.add_argument("--config", required=True, help="Path to eval.yaml")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--run-id", help="Run ID (default: timestamp)")
    parser.add_argument("-n", "--parallelism", type=int, default=1)
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="Keep sandboxes after trial (or set AGENT_EVAL_OPENSHELL_KEEP_RUN=1)",
    )
    parser.add_argument("--cases", nargs="+", help="Case IDs to run (default: all)")
    parser.add_argument(
        "--no-llm-judges", action="store_true", help="Skip LLM judges"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    keep = args.keep_sandbox or os.environ.get("AGENT_EVAL_OPENSHELL_KEEP_RUN") == "1"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    exit_code = asyncio.run(
        run_openshell(
            config_path=Path(args.config),
            model=args.model,
            run_id=run_id,
            parallelism=args.parallelism,
            keep_sandbox=keep,
            cases=args.cases,
            no_llm_judges=args.no_llm_judges,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
