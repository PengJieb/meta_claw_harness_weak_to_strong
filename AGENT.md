# MetaClaw Agent Guide

This file is for coding agents working in this repository. Read it before changing code so you do not have to rediscover the project shape and local conventions.

## Project Overview

MetaClaw is a Python 3.10+ package that runs an OpenAI-compatible proxy in front of personal agents such as OpenClaw. The proxy can inject skills, persist/retrieve long-term memory, collect conversation samples, score them, and trigger RL updates through Tinker-compatible training backends. The public CLI entry point is `metaclaw`.

The repository also contains:

- `metaclaw/`: main Python package.
- `metaclaw/memory/`: local-first long-term memory subsystem.
- `tests/`: pytest test suite for CLI, setup/config, runtime state, memory, SDK backend, launcher, and utilities.
- `examples/`: small scripts for conversation replay/RL experiments.
- `scripts/`: ad hoc benchmark and experiment runners.
- `benchmark/`: separate `metaclaw-bench` Python package and dataset validation/inference/scoring/reporting utilities.
- `extensions/metaclaw-openclaw/`: OpenClaw extension wrapper.
- `openclaw-metaclaw-memory/`: TypeScript OpenClaw memory plugin plus Python sidecar.

## Core Architecture

`metaclaw.cli` defines the `metaclaw` Click command group. Important commands include:

- `metaclaw setup`: runs the interactive setup wizard.
- `metaclaw start`: starts the proxy and, depending on mode, the training/scheduler stack.
- `metaclaw stop`: stops a running instance by PID file.
- `metaclaw status`: checks the PID file and `/healthz`.
- `metaclaw train-step`: triggers a manual RL update through the admin API.
- `metaclaw config`: reads/writes user config dotpaths.
- `metaclaw uninstall`: removes local MetaClaw state and extension files.

`metaclaw.config.MetaClawConfig` is the internal dataclass used by runtime components. `metaclaw.config_store.ConfigStore` is the user-facing YAML store at `~/.metaclaw/config.yaml`; it deep-merges defaults and bridges nested YAML sections into `MetaClawConfig`. Prefer adding new user-facing settings to both `_DEFAULTS` and `to_metaclaw_config()`.

`metaclaw.launcher.MetaClawLauncher` orchestrates runtime mode:

- `skills_only`: proxy, skill injection, optional skill evolution, optional memory.
- `rl`: full training stack without scheduler gating.
- `auto`: RL with scheduler-gated slow updates.

`metaclaw.api_server.MetaClawAPIServer` is the FastAPI proxy. It normalizes chat payloads, handles OpenAI/Anthropic-compatible formats, injects skills/memory, forwards to the upstream LLM or sampling client, records samples, exposes health/admin endpoints, and coordinates memory ingestion.

`metaclaw.skill_manager.SkillManager` loads Claude-style skills from `*/SKILL.md` files with YAML frontmatter. It supports keyword/template retrieval and optional embedding retrieval.

`metaclaw.memory` provides structured long-term memory:

- `models.py`: dataclasses/enums such as `MemoryUnit`, `MemoryType`, `MemoryQuery`.
- `store.py`: SQLite persistence/search.
- `manager.py`: high-level retrieval, rendering, session buffering, extraction, consolidation.
- `policy*.py`, `telemetry.py`, `upgrade_worker.py`: adaptive memory policy and background upgrade support.

The benchmark package under `benchmark/` is intentionally separate. Run it from `benchmark/` when installing it, but dataset paths in docs often assume the repository root.

## Local Commands

Local machines are for coding and development only. Do not run formal experiments locally, and do not run experiment smoke tests locally. Keep local execution limited to lightweight checks that make sure the code is syntactically and structurally sane before syncing to the server.

Use conda for local development environments:

```bash
conda create -n metaclaw-dev python=3.10
conda activate metaclaw-dev
```

Install the main package for development:

```bash
pip install -e .
pip install -e ".[evolve,scheduler]"  # optional non-GPU development extras when needed
```

Allowed local checks are lightweight only:

```bash
python -m compileall metaclaw
ruff check .
pytest tests/test_config_store.py tests/test_utils.py -q
```

Do not use local hardware for GPU, RL, benchmark, full integration, end-to-end, live LLM, or smoke-test experiment runs. Those belong on the server.

## Server Workflow

Formal tests, smoke tests, benchmark runs, RL jobs, GPU workloads, and experiment runs must be done on the server, not locally.

SSH host config:

```sshconfig
Host jump.pjlab.org.cn
  HostName jump.pjlab.org.cn
  User pengjie@pengjie@10.140.37.164
  HostKeyAlgorithms +ssh-rsa
  PubkeyAcceptedKeyTypes +ssh-rsa
```

SSH login goes through a proxy server and may take a while. The normal interactive path is:

```bash
ssh jump.pjlab.org.cn
```

The login may print a line like `开始连接到 pengjie@10.140.37.164`, then show the cluster login banner. Do not assume the connection failed just because there is no output for the first minute.

Important: the cluster management/login node must not be used for data copy, migration, experiments, smoke tests, or other heavy work. Use it only to log in, inspect lightweight files, submit Slurm jobs, and monitor job state.

Always use SFTP for file transfer between local and server. Do not use the cluster management/login node to run data copy or migration commands such as `scp`, `rsync`, large `cp`, or bulk archive extraction. For SFTP, the home folder path is:

```text
/Default/10.140.37.162
```

On the server, keep this repository under:

```text
~/MetaClaw
```

Use conda on the server as well. The server conda environment is the one used for experiment dependencies, GPU libraries, vLLM, and benchmark/RL execution.

Server jobs that need GPU must be provisioned with Slurm. Before creating or changing an `sbatch` script, read this server template:

```bash
ssh jump.pjlab.org.cn 'sed -n "1,240p" ~/slurm_scripts_template/sbatch_template.sh'
```

Experiments that require an LLM API should launch a local model API with vLLM on the allocated server node. Use this server script as the reference pattern:

```bash
ssh jump.pjlab.org.cn 'sed -n "1,240p" ~/XSkill/model/vllm_server_qwen3_vl_8chat.sh'
```

Known server paths for the Qwen3.5-9B skills-only benchmark:

- Model checkpoint: `~/XSkill/model/Qwen3.5-9B`
- vLLM Apptainer image: `~/XSkill/model/vllm-openai-v0.21.0-cu129.sif`
- MetaClaw checkout: `~/MetaClaw`
- Slurm smoke script used during validation: `~/metaclaw_smoke_qwen35_skills_only.sbatch`

Current server environment notes:

- `unlearning_skill` is the working conda env used for the Qwen3.5-9B skills-only smoke run. It already has `torch`, `openai`, `tiktoken`, and `numpy`; install/check `fastapi`, `uvicorn[standard]`, `httpx`, `click`, and `pyyaml` before benchmark runs.
- `ct_process` has the web stack but lacks `torch`, `openai`, `numpy`, and `tiktoken`, so it is not enough for `metaclaw.api_server` without extra installs.
- OpenClaw `2026.6.1` requires Node `>=22.19`. The server's old Node 18 is insufficient, and the official Node 22 binary may fail on the server glibc. The working fallback is the unofficial `node-v22.19.0-linux-x64-glibc-217` build installed under `~/.cache/metaclaw-node/`.
- Server proxy variables may be enabled globally. Before probing localhost services such as vLLM, MetaClaw proxy, or OpenClaw gateway, set `no_proxy` and `NO_PROXY` to include `127.0.0.1,localhost,0.0.0.0`; otherwise localhost health checks can go through the proxy and return misleading `503` responses.
- A completed infrastructure smoke test was run through Slurm job `9672810` on 2026-06-05. It launched vLLM/Qwen3.5-9B, started the MetaClaw proxy in `skills_only` mode, started an OpenClaw gateway, executed one small benchmark scene, and wrote results to `~/MetaClaw/benchmark/results/run_20260605_161248`. The run completed mechanically with 0% task accuracy, so treat it as an infrastructure smoke pass rather than a quality/performance result.

For higher-throughput benchmark runs, use the eight-replica vLLM router:

- Router script: `benchmark/scripts/vllm_batch_router.py`
- Fleet launcher: `benchmark/scripts/launch_vllm_fleet_then_run.py`
- Slurm entrypoint: `benchmark/scripts/run_skills_only_qwen35_9b_vllm_fleet.sbatch`

The fleet launcher starts one vLLM server per GPU with `CUDA_VISIBLE_DEVICES` pinned per process, waits for all `/health` endpoints, starts a local OpenAI-compatible batching router, and runs the benchmark with `BENCHMARK_BASE_URL=http://127.0.0.1:${VLLM_ROUTER_PORT}/v1`. The router batches non-streaming requests for `VLLM_ROUTER_BATCH_WINDOW` seconds, then fans them out across the vLLM replicas. It intentionally rejects `stream=true` because streaming responses cannot be safely held and batch-dispatched. The Slurm script generates two per-job API keys by default: `VLLM_BACKEND_API_KEY` for wrapper-to-vLLM calls and `VLLM_ROUTER_API_KEY` for MetaClaw-to-wrapper calls. Do not use `EMPTY` for fleet runs; logs are created under `umask 077`.

If the server cannot access the network, run the `proxy_on` alias from the server `~/.bashrc` before installing dependencies, pulling packages, or accessing remote model/API resources.

Use the Codex skill `$metaclaw-server-jobs` whenever provisioning a Slurm job, syncing code for a server experiment, or launching a local vLLM API server for MetaClaw experiments.

## TypeScript Plugin Commands

Build the TypeScript memory plugin:

```bash
cd openclaw-metaclaw-memory
npm install
npm run build
```

Install/run the benchmark package:

```bash
cd benchmark
pip install -e ".[dev]"
pytest
metaclaw-bench check -p data/metaclaw-bench/all_tests.json
```

Run benchmark commands only on the server unless the task is purely inspecting CLI behavior without executing an experiment.

Some tests may be ahead of the current implementation. If a test references a helper that is missing from the code under test, inspect the test carefully and decide whether the intended behavior should be implemented or the test is stale.

## Python Coding Style

Follow the existing Python style:

- Target Python `>=3.10`.
- Use `from __future__ import annotations` in modules that use modern type annotations or forward references.
- Prefer dataclasses for structured configuration and domain records.
- Keep type hints practical: use `str | None`, `list[dict]`, `dict[str, Any]` in newer code; older modules may still use `Optional`, `List`, and `Dict`.
- Use module-level `logger = logging.getLogger(__name__)` for runtime diagnostics.
- Use Click for CLI commands and `click.echo()` for user-facing CLI output.
- Use FastAPI primitives in API code: raise `HTTPException` for request errors and return `JSONResponse`/`StreamingResponse` when needed.
- Prefer small helper functions near the code path they support. Many modules use section banners like `# ------------------------------------------------------------------ #` to group helpers and lifecycle methods.
- Preserve backward-compatible aliases and config keys unless you are intentionally migrating them. This project has users with existing `~/.metaclaw/config.yaml` files.
- Treat network providers and optional dependencies as best-effort. Catch import/config errors where optional features should degrade gracefully.
- Do not log or print API keys, OAuth tokens, bearer tokens, or raw secrets.

Ruff is configured in `pyproject.toml` with:

- line length: `120`
- selected rules: `E`, `F`, `I`
- ignored rules: `E731`, `F403`, `F405`

Imports should be sorted by Ruff/isort. The existing code often uses local imports inside CLI handlers or optional feature branches to avoid requiring optional dependencies at import time; keep that pattern.

## Async, Threads, and Lifecycle

MetaClaw mixes asyncio, threads, and blocking subprocesses. Be conservative:

- Do not block the event loop with long synchronous work inside async paths.
- Keep proxy serving, scheduler triggers, training, and memory upgrade workers decoupled.
- When adding background tasks, make cancellation/stop paths explicit.
- PID files live under `~/.metaclaw` and are keyed by proxy port.
- `/healthz` is used by CLI/status/startup checks; keep it cheap and dependency-light.

## Config Conventions

User config is nested YAML, but runtime config is flat `MetaClawConfig`.

Common user-facing sections:

- `mode`: `auto`, `rl`, or `skills_only`.
- `llm`: upstream LLM provider/auth/base/model settings.
- `proxy`: host/port.
- `skills`: skill injection and evolution settings.
- `rl`: training backend, PRM, LoRA, batch, and manual trigger settings.
- `memory`: store path, scope, retrieval mode, policy limits, extraction/consolidation settings.
- `scheduler`: idle/sleep/calendar scheduling.
- `wechat`: official OpenClaw WeChat plugin toggle.

When adding a config option:

1. Add a default in `ConfigStore._DEFAULTS`.
2. Add a field in `MetaClawConfig` if runtime components need it.
3. Map it in `ConfigStore.to_metaclaw_config()`.
4. Update setup/config tests or add a focused test.
5. Keep old key aliases working when practical.

## Memory Conventions

Memory data is local and scope-aware. Use `scope_id` consistently and avoid cross-scope reads/writes unless the feature explicitly requires it.

Memory units have one of these types:

- `episodic`
- `semantic`
- `preference`
- `project_state`
- `working_summary`
- `procedural_observation`

Prefer using `MemoryManager` instead of reaching directly into `MemoryStore` from unrelated code. `MemoryManager` handles extraction, buffering, rendering, policy, consolidation, telemetry, and retrieval cache behavior.

Session ingestion supports incremental buffering through `buffer_turn()` and final flushing through `flush_session(final=True)`. If you change turn capture behavior, test both mid-session flushes and final session summaries.

## Skill Conventions

Skills are directory-based:

```text
skills_dir/
  skill-name/
    SKILL.md
```

`SKILL.md` requires YAML frontmatter with `name` and `description`; `category` defaults to `general`. Valid categories are documented in `metaclaw/skill_manager.py`. Keep parsing simple and compatible with existing frontmatter.

Skill evolution can mutate the skill directory and increments `SkillManager.generation`. Training code relies on that generation to avoid mixing stale samples with post-evolution behavior.

## TypeScript Plugin Style

The `openclaw-metaclaw-memory/` package is ESM TypeScript:

- `type: "module"`
- `module`/`moduleResolution`: `NodeNext`
- target: `ES2022`
- strict mode enabled
- Node `>=18`

Use extension-bearing relative imports such as `./types.js` in source files because the package emits NodeNext ESM. Keep plugin config defaults in `src/types.ts` aligned with `config-schema.ts` and `openclaw.plugin.json`.

The plugin registers a managed Python sidecar service, then exposes hooks, AI tools, slash commands, and CLI commands. Do not make hook/tool code assume the sidecar client is initialized before service startup; use the existing lazy `getClient` pattern.

## Testing Guidance

Prefer focused tests close to the behavior you change:

- CLI behavior: `click.testing.CliRunner` with monkeypatching.
- Memory behavior: temporary directories and real `MemoryStore`/`MemoryManager`.
- Async API/proxy behavior: `asyncio.run()` in unit tests or pytest async support where already used.
- Optional integrations: monkeypatch provider clients/subprocesses instead of requiring real external services.
- Benchmark package: run tests from `benchmark/` or with its `pythonpath` settings.

Avoid tests that depend on the user's real home directory, OpenClaw installation, API keys, calendar credentials, or live LLM calls unless the filename/marker clearly indicates a live integration test.

## Safety Notes

- This repo handles credentials for LLM providers, OAuth-backed CLI providers, Tinker/MinT/Weaver, Bedrock, and Google Calendar. Never persist secrets outside the intended config/auth store.
- Be careful with `metaclaw uninstall`; it removes user state under `~/.metaclaw` and extension files.
- Do not change benchmark datasets or generated memory/record data unless the task is specifically about those artifacts.
- Preserve API compatibility for OpenAI-compatible `/v1/chat/completions` clients and Anthropic-compatible `/v1/messages` clients when editing proxy request/response normalization.
