#!/usr/bin/env bash
#
# Start the Qwen3.5-9B vLLM fleet inside an existing Slurm GPU allocation.
#
# Expected use:
#   salloc -p moe_p --quotatype=reserved -N 1 --gres=gpu:8 -c 64 --mem=512G -t 08:00:00
#   srun --pty bash
#   cd ~/MetaClaw
#   bash benchmark/scripts/start_qwen35_9b_vllm_fleet.sh
#
# Keep the allocation alive while clients/eval jobs call the printed router URL.

set -eo pipefail
umask 077

export METACLAW_ROOT="${METACLAW_ROOT:-$HOME/MetaClaw}"
export METACLAW_CONDA_ENV="${METACLAW_CONDA_ENV:-unlearning_skill}"
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$HOME/XSkill/model/Qwen3.5-9B}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen3.5-9B}"
export VLLM_SIF_PATH="${VLLM_SIF_PATH:-$HOME/XSkill/model/vllm-openai-v0.21.0-cu129.sif}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-18000}"
export VLLM_ROUTER_HOST="${VLLM_ROUTER_HOST:-0.0.0.0}"
export VLLM_ROUTER_PORT="${VLLM_ROUTER_PORT:-19000}"
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

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_NO_SLURM:-0}" != "1" ]]; then
  echo "[start] ERROR: no SLURM_JOB_ID detected." >&2
  echo "[start] Allocate GPUs with salloc and run this script on the allocated GPU node." >&2
  exit 2
fi

mkdir -p "$VLLM_LOGS_DIR"
cd "$METACLAW_ROOT"

if [[ ! -d "$VLLM_MODEL_PATH" ]]; then
  echo "[start] ERROR: VLLM_MODEL_PATH does not exist: $VLLM_MODEL_PATH" >&2
  exit 2
fi
if [[ -n "$VLLM_SIF_PATH" && ! -f "$VLLM_SIF_PATH" ]]; then
  echo "[start] ERROR: VLLM_SIF_PATH does not exist: $VLLM_SIF_PATH" >&2
  exit 2
fi
if [[ -n "$VLLM_SIF_PATH" ]] && ! command -v apptainer >/dev/null 2>&1; then
  echo "[start] ERROR: apptainer not found but VLLM_SIF_PATH is set." >&2
  exit 2
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
  echo "[start] ERROR: conda not found; set up conda before launching vLLM." >&2
  exit 2
fi

python - <<'PY'
import importlib.util
import sys

required = ["fastapi", "httpx", "uvicorn"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[start] ERROR: missing Python modules: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(2)
PY

python benchmark/scripts/start_vllm_fleet.py \
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
  "$@"
