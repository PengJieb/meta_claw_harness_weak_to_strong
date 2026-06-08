#!/usr/bin/bash
#
# CPU-only Slurm entrypoint for MetaClaw Benchmark skills_only mode against an
# already-running manual vLLM fleet/router.
#
# Start the GPU fleet first and keep its Slurm step alive, for example:
#   salloc -p moe_p --quotatype=reserved -N 1 --gres=gpu:8 -c 64 --mem=512G -t 08:00:00
#   srun --pty bash
#   cd ~/MetaClaw
#   bash benchmark/scripts/start_qwen35_9b_vllm_fleet.sh
#   # Leave this shell running while the benchmark job uses the router.
#
# Then submit this CPU-only benchmark job from the login node:
#   cd ~/MetaClaw
#   VLLM_FLEET_STATE_FILE=benchmark/logs/vllm_fleet/vllm_fleet_state_<jobid>.json \
#     sbatch benchmark/scripts/run_skills_only_qwen35_9b_existing_vllm.sbatch

#SBATCH --job-name=mc-skill-api
#SBATCH --output=benchmark/logs/slurm/skills_only_existing_vllm_%j.log
#SBATCH --error=benchmark/logs/slurm/skills_only_existing_vllm_%j.log
#SBATCH --partition=moe_p
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --quotatype=reserved
#SBATCH --time=08:00:00

set -eo pipefail
umask 077

export METACLAW_ROOT="${METACLAW_ROOT:-$HOME/MetaClaw}"
export METACLAW_CONDA_ENV="${METACLAW_CONDA_ENV:-unlearning_skill}"
export METACLAW_NODE_VERSION="${METACLAW_NODE_VERSION:-22.19.0}"
export METACLAW_NODE_HOME="${METACLAW_NODE_HOME:-$HOME/.cache/metaclaw-node/node-v${METACLAW_NODE_VERSION}-linux-x64-glibc-217}"
export VLLM_LOGS_DIR="${VLLM_LOGS_DIR:-$METACLAW_ROOT/benchmark/logs/vllm_fleet}"
export METACLAW_SKILLS_DIR="${METACLAW_SKILLS_DIR:-$METACLAW_ROOT/memory_data/skills}"
export METACLAW_BENCH_INPUT="${METACLAW_BENCH_INPUT:-$METACLAW_ROOT/benchmark/data/metaclaw-bench/all_tests_metaclaw.json}"
export METACLAW_BENCH_BIN="${METACLAW_BENCH_BIN:-metaclaw-bench}"
export METACLAW_BIN="${METACLAW_BIN:-metaclaw}"
unset METACLAW_API_KEY_SCRIPT

mkdir -p "$METACLAW_ROOT/benchmark/logs/slurm"
cd "$METACLAW_ROOT"

if [[ -z "${VLLM_FLEET_STATE_FILE:-}" ]]; then
  VLLM_FLEET_STATE_FILE="$(ls -t "$VLLM_LOGS_DIR"/vllm_fleet_state_*.json 2>/dev/null | head -1 || true)"
fi
if [[ -z "$VLLM_FLEET_STATE_FILE" || ! -f "$VLLM_FLEET_STATE_FILE" ]]; then
  echo "[job] ERROR: VLLM_FLEET_STATE_FILE not found: ${VLLM_FLEET_STATE_FILE:-<empty>}" >&2
  exit 2
fi
export VLLM_FLEET_STATE_FILE

echo "[job] host=$(hostname)"
echo "[job] time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[job] METACLAW_ROOT=$METACLAW_ROOT"
echo "[job] METACLAW_CONDA_ENV=$METACLAW_CONDA_ENV"
echo "[job] VLLM_FLEET_STATE_FILE=$VLLM_FLEET_STATE_FILE"
echo "[job] METACLAW_SKILLS_DIR=$METACLAW_SKILLS_DIR"
echo "[job] METACLAW_BENCH_INPUT=$METACLAW_BENCH_INPUT"

if [[ ! -d "$METACLAW_SKILLS_DIR" ]]; then
  echo "[job] ERROR: METACLAW_SKILLS_DIR does not exist: $METACLAW_SKILLS_DIR" >&2
  exit 2
fi

if [[ -x "$METACLAW_NODE_HOME/bin/node" ]]; then
  export PATH="$METACLAW_NODE_HOME/bin:$PATH"
fi
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$METACLAW_CONDA_ENV"
else
  echo "[job] ERROR: conda not found; set up conda before submitting." >&2
  exit 2
fi

find_openclaw_bin() {
  local candidate
  for candidate in \
    "${OPENCLAW_BIN:-}" \
    "$METACLAW_NODE_HOME/bin/openclaw" \
    "$(command -v openclaw 2>/dev/null || true)" \
    "$HOME/.local/bin/openclaw" \
    "$HOME/.npm-global/bin/openclaw"
  do
    if [[ -n "$candidate" && -x "$candidate" && ! -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! OPENCLAW_BIN="$(find_openclaw_bin)"; then
  echo "[job] ERROR: openclaw executable not found." >&2
  exit 2
fi
export OPENCLAW_BIN
echo "[job] OPENCLAW_BIN=$OPENCLAW_BIN"
"$OPENCLAW_BIN" --version || {
  echo "[job] ERROR: OPENCLAW_BIN failed --version: $OPENCLAW_BIN" >&2
  exit 2
}

benchmark_env_exports="$(
python - <<'PY'
import json
import os
import shlex
import urllib.request
from pathlib import Path

state_path = Path(os.environ["VLLM_FLEET_STATE_FILE"]).expanduser()
state = json.loads(state_path.read_text(encoding="utf-8"))
router_url = state["router_url"].rstrip("/")
api_key = state["router_api_key"]
model = state.get("model_name") or "Qwen3.5-9B"

host = router_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
for key in ("no_proxy", "NO_PROXY"):
    current = os.environ.get(key, "")
    values = [item.strip() for item in current.split(",") if item.strip()]
    if host and host not in values:
        values.append(host)
    value = ",".join(values)
    os.environ[key] = value
    print(f"export {key}={shlex.quote(value)}")

req = urllib.request.Request(
    router_url.replace("/v1", "/health"),
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(req, timeout=10) as resp:
    if resp.status != 200:
        raise SystemExit(f"router health returned HTTP {resp.status}")

print(f"export BENCHMARK_BASE_URL={shlex.quote(router_url)}")
print(f"export BENCHMARK_API_KEY={shlex.quote(api_key)}")
print(f"export BENCHMARK_MODEL={shlex.quote(model)}")
PY
)"
eval "$benchmark_env_exports"
echo "[job] BENCHMARK_BASE_URL=$BENCHMARK_BASE_URL"
echo "[job] BENCHMARK_MODEL=$BENCHMARK_MODEL"
echo "[job] BENCHMARK_API_KEY=<set>"

python - <<'PY'
import importlib.util
import sys

required = ["yaml", "fastapi", "uvicorn", "httpx", "openai", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[job] ERROR: missing Python modules: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(2)
PY

python benchmark/scripts/skills_only_run.py
