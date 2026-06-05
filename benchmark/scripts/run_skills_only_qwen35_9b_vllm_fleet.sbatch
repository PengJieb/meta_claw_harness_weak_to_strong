#!/usr/bin/bash
#
# Slurm entrypoint for MetaClaw Benchmark skills_only mode with eight
# one-GPU vLLM replicas behind a local batching router.
#
# Submit from the server login node:
#   cd ~/MetaClaw
#   sbatch benchmark/scripts/run_skills_only_qwen35_9b_vllm_fleet.sbatch
#
# The login node is only for submission/monitoring. vLLM, the router, and the
# benchmark run inside the Slurm allocation.

#SBATCH --job-name=mc-skill-q35-8x
#SBATCH --output=benchmark/logs/slurm/skills_only_qwen35_9b_fleet_%j.log
#SBATCH --error=benchmark/logs/slurm/skills_only_qwen35_9b_fleet_%j.log
#SBATCH --partition=moe_p
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH --time=08:00:00

set -eo pipefail
umask 077

export METACLAW_ROOT="${METACLAW_ROOT:-$HOME/MetaClaw}"
export METACLAW_CONDA_ENV="${METACLAW_CONDA_ENV:-unlearning_skill}"
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$HOME/XSkill/model/Qwen3.5-9B}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen3.5-9B}"
export VLLM_SIF_PATH="${VLLM_SIF_PATH:-$HOME/XSkill/model/vllm-openai-v0.21.0-cu129.sif}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-18000}"
export VLLM_ROUTER_HOST="${VLLM_ROUTER_HOST:-127.0.0.1}"
export VLLM_ROUTER_PORT="${VLLM_ROUTER_PORT:-19000}"
make_api_key() {
  python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
}

export VLLM_BACKEND_API_KEY="${VLLM_BACKEND_API_KEY:-${VLLM_API_KEY:-}}"
if [[ -z "$VLLM_BACKEND_API_KEY" || "$VLLM_BACKEND_API_KEY" == "EMPTY" ]]; then
  export VLLM_BACKEND_API_KEY="$(make_api_key)"
fi
export VLLM_ROUTER_API_KEY="${VLLM_ROUTER_API_KEY:-}"
if [[ -z "$VLLM_ROUTER_API_KEY" || "$VLLM_ROUTER_API_KEY" == "EMPTY" ]]; then
  export VLLM_ROUTER_API_KEY="$(make_api_key)"
fi
export VLLM_API_KEY="$VLLM_BACKEND_API_KEY"
export VLLM_NUM_SERVERS="${VLLM_NUM_SERVERS:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_xml}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
export VLLM_ROUTER_BATCH_WINDOW="${VLLM_ROUTER_BATCH_WINDOW:-0.25}"
export VLLM_ROUTER_MAX_BATCH_SIZE="${VLLM_ROUTER_MAX_BATCH_SIZE:-64}"
export VLLM_ROUTER_REQUEST_TIMEOUT="${VLLM_ROUTER_REQUEST_TIMEOUT:-900}"
export VLLM_LOGS_DIR="${VLLM_LOGS_DIR:-$METACLAW_ROOT/benchmark/logs/vllm_fleet}"
export METACLAW_SKILLS_DIR="${METACLAW_SKILLS_DIR:-$METACLAW_ROOT/memory_data/skills}"
export METACLAW_BENCH_INPUT="${METACLAW_BENCH_INPUT:-$METACLAW_ROOT/benchmark/data/metaclaw-bench/all_tests_metaclaw.json}"
export METACLAW_BENCH_BIN="${METACLAW_BENCH_BIN:-metaclaw-bench}"
export METACLAW_BIN="${METACLAW_BIN:-metaclaw}"
unset METACLAW_API_KEY_SCRIPT

mkdir -p "$METACLAW_ROOT/benchmark/logs/slurm" "$VLLM_LOGS_DIR"
cd "$METACLAW_ROOT"

echo "[job] host=$(hostname)"
echo "[job] time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[job] METACLAW_ROOT=$METACLAW_ROOT"
echo "[job] METACLAW_CONDA_ENV=$METACLAW_CONDA_ENV"
echo "[job] VLLM_MODEL_PATH=$VLLM_MODEL_PATH"
echo "[job] VLLM_MODEL_NAME=$VLLM_MODEL_NAME"
echo "[job] VLLM_SIF_PATH=$VLLM_SIF_PATH"
echo "[job] VLLM_NUM_SERVERS=$VLLM_NUM_SERVERS"
echo "[job] VLLM_BASE_PORT=$VLLM_BASE_PORT"
echo "[job] VLLM_ROUTER_PORT=$VLLM_ROUTER_PORT"
echo "[job] VLLM_BACKEND_API_KEY=<set>"
echo "[job] VLLM_ROUTER_API_KEY=<set>"
echo "[job] METACLAW_SKILLS_DIR=$METACLAW_SKILLS_DIR"
echo "[job] METACLAW_BENCH_INPUT=$METACLAW_BENCH_INPUT"

if [[ ! -d "$VLLM_MODEL_PATH" ]]; then
  echo "[job] ERROR: VLLM_MODEL_PATH does not exist: $VLLM_MODEL_PATH" >&2
  exit 2
fi

if [[ -n "$VLLM_SIF_PATH" && ! -f "$VLLM_SIF_PATH" ]]; then
  echo "[job] ERROR: VLLM_SIF_PATH does not exist: $VLLM_SIF_PATH" >&2
  exit 2
fi

if [[ -n "$VLLM_SIF_PATH" ]] && ! command -v apptainer >/dev/null 2>&1; then
  echo "[job] ERROR: apptainer not found but VLLM_SIF_PATH is set." >&2
  exit 2
fi

if [[ ! -d "$METACLAW_SKILLS_DIR" ]]; then
  echo "[job] ERROR: METACLAW_SKILLS_DIR does not exist: $METACLAW_SKILLS_DIR" >&2
  exit 2
fi

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
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

python - <<'PY'
import importlib.util
import sys

required = ["yaml", "fastapi", "uvicorn", "httpx", "openai", "torch"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[job] ERROR: missing Python modules: {', '.join(missing)}", file=sys.stderr)
    print("[job] If server network is unavailable, run proxy_on before installing dependencies.", file=sys.stderr)
    raise SystemExit(2)
PY

python benchmark/scripts/launch_vllm_fleet_then_run.py \
  --model-path "$VLLM_MODEL_PATH" \
  --model-name "$VLLM_MODEL_NAME" \
  --sif-path "$VLLM_SIF_PATH" \
  --vllm-host "$VLLM_HOST" \
  --base-port "$VLLM_BASE_PORT" \
  --router-host "$VLLM_ROUTER_HOST" \
  --router-port "$VLLM_ROUTER_PORT" \
  --num-servers "$VLLM_NUM_SERVERS" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --dtype "$VLLM_DTYPE" \
  --reasoning-parser "$VLLM_REASONING_PARSER" \
  --tool-call-parser "$VLLM_TOOL_CALL_PARSER" \
  --startup-timeout "$VLLM_STARTUP_TIMEOUT" \
  --logs-dir "$VLLM_LOGS_DIR" \
  --router-batch-window "$VLLM_ROUTER_BATCH_WINDOW" \
  --router-max-batch-size "$VLLM_ROUTER_MAX_BATCH_SIZE" \
  --router-request-timeout "$VLLM_ROUTER_REQUEST_TIMEOUT" \
  -- \
  python benchmark/scripts/skills_only_run.py
