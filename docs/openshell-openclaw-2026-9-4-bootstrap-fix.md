# OpenClaw 2026.9.4 database bootstrap compatibility

## Why this branch exists

The `saw-mpk` PipelineRun `sana-morning-briefing-main-e4b2ffc-qm9pj` used
`GuyZivRH/agent-eval-harness` at `feat/aeh-openshell-openclaw` and the Chief of
Staff image `ghcr.io/sanafayyaz315/openclaw-saw-agent@sha256:042f8a3434fed0b59b21340f8ea9392f2e011aa2f8d924ba858ec88e0c934ab6`.
Both cases failed before agent execution with `Expected one image database
initializer` from `agent_eval/openshell/forge_bootstrap.mjs`.

The harness searched only for `openclaw-agent-db-maintenance-*.js`. The pinned
image contains OpenClaw `2026.9.4`, whose initializer is exported by
`openclaw-agent-db-*.mjs`. The patch searches both layouts and selects the
single named `ensureOpenClawAgentDatabaseSchema` export when available. It
retains the older function-name fallback.

## Repository and branch setup

The authenticated GitHub account was `sanafayyaz315`. It had read access but
no push access to Guy's fork, so a personal fork was created. The local clone
is `/Users/sanafayyaz/Desktop/Red Hat/agent-eval-harness-sana`.

```sh
gh repo fork GuyZivRH/agent-eval-harness --clone=false
gh repo clone sanafayyaz315/agent-eval-harness \
  '/Users/sanafayyaz/Desktop/Red Hat/agent-eval-harness-sana'
cd '/Users/sanafayyaz/Desktop/Red Hat/agent-eval-harness-sana'
git fetch upstream feat/aeh-openshell-openclaw
git switch -c fix/openclaw-2026-9-4-db-bootstrap \
  upstream/feat/aeh-openshell-openclaw
```

`origin` is `sanafayyaz315/agent-eval-harness`; `upstream` is
`GuyZivRH/agent-eval-harness`. The branch was created from Guy's
`feat/aeh-openshell-openclaw` at
`fbc9a33256affc4d245e6080e2fac493ace87786`. It was not based on either
repository's `main` branch.

## Verification and next evaluation

Run `node --check agent_eval/openshell/forge_bootstrap.mjs` and `git diff --check`.
Then run the bootstrap in a disposable container from the pinned image:

```sh
podman run --rm -i --platform linux/amd64 --entrypoint node \
  'ghcr.io/sanafayyaz315/openclaw-saw-agent:main-sha-e4b2ffc0f099d07249e6f314f9a270185320e389' \
  --input-type=module < agent_eval/openshell/forge_bootstrap.mjs
```

Success prints the four `IMAGE_FILE_READABLE` checks and
`FORGE_IMAGE_WORKSPACE_OK`. For the next PipelineRun, set
`agent-eval-harness-repo-url` to
`https://github.com/sanafayyaz315/agent-eval-harness.git` and
`agent-eval-harness-repo-revision` to this fix branch. Keep the image digest
and evaluation definitions unchanged for a controlled rerun. Do not interpret
judge scores from the failed PipelineRun as skill quality; neither case reached
agent execution. Its cleanup TaskRun also failed independently because
`registry.redhat.io/openshift4/ose-cli:latest` is unsupported.

## First CI rerun and follow-up

The first rerun was `sana-morning-briefing-main-harness-fix-ktm8d`. Both
cases logged `IMAGE_FILE_READABLE` and `FORGE_IMAGE_WORKSPACE_OK`, proving
that the database initializer fix works in `saw-mpk`. The next preflight
returned HTTP 200 with 128 completion tokens, all of them reasoning tokens,
and an empty content field. No agent execution occurred.

This branch therefore also raises the OpenShell GLM preflight request from
128 to 512 completion tokens, matching the existing
`mpk/flash-preflight-512` harness variant. The test asserts that the
preflight sends 512. The SAW image and evaluation definitions remain unchanged.
