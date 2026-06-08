#!/usr/bin/env bash
#
# Stop a Qwen3.5-9B vLLM fleet started by start_qwen35_9b_vllm_fleet.sh.
# Run this on the same allocated GPU node before releasing the salloc session.

set -eo pipefail

export METACLAW_ROOT="${METACLAW_ROOT:-$HOME/MetaClaw}"
export METACLAW_CONDA_ENV="${METACLAW_CONDA_ENV:-unlearning_skill}"
export VLLM_LOGS_DIR="${VLLM_LOGS_DIR:-$METACLAW_ROOT/benchmark/logs/vllm_fleet}"

cd "$METACLAW_ROOT"

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
fi

python benchmark/scripts/stop_vllm_fleet.py --logs-dir "$VLLM_LOGS_DIR" "$@"
