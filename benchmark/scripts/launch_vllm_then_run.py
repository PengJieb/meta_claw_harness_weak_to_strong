#!/usr/bin/env python3
"""Launch a vLLM OpenAI-compatible server, wait for readiness, run a command.

This helper is intended for Slurm jobs where the benchmark client and vLLM
server run on the same allocated node.  It owns vLLM cleanup so failed
benchmark runs do not leave a model server behind inside the allocation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _expand(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _default_tp_size() -> int:
    for key in ("VLLM_TENSOR_PARALLEL_SIZE", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            # SLURM_GPUS_PER_NODE can be values like "8" or "gpu:8".
            return int(str(raw).split(":")[-1])
        except ValueError:
            pass
    return 1


def _build_vllm_cmd(args: argparse.Namespace) -> list[str]:
    server_cmd = [
        "python3" if args.sif_path else sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    model_path = _expand(args.model_path)
    if args.sif_path:
        server_cmd += ["--model", "/model_weights"]
    else:
        server_cmd += ["--model", model_path]
    server_cmd += [
        "--served-model-name",
        args.model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len:
        server_cmd += ["--max-model-len", str(args.max_model_len)]
    if args.dtype:
        server_cmd += ["--dtype", args.dtype]
    if args.trust_remote_code:
        server_cmd.append("--trust-remote-code")
    if args.api_key:
        server_cmd += ["--api-key", args.api_key]
    if args.reasoning_parser:
        server_cmd += ["--reasoning-parser", args.reasoning_parser]
    if args.enforce_eager:
        server_cmd.append("--enforce-eager")
    if args.enable_auto_tool_choice:
        server_cmd.append("--enable-auto-tool-choice")
    if args.tool_call_parser:
        server_cmd += ["--tool-call-parser", args.tool_call_parser]
    if args.extra_args:
        server_cmd += shlex.split(args.extra_args)
    if not args.sif_path:
        return server_cmd
    return [
        "apptainer",
        "exec",
        "--nv",
        "--bind",
        f"{model_path}:/model_weights",
        _expand(args.sif_path),
        *server_cmd,
    ]


def _ready(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_vllm(proc: subprocess.Popen, base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    health_url = base_url.rstrip("/") + "/health"
    models_url = base_url.rstrip("/") + "/v1/models"
    print(f"[vllm] Waiting for readiness: {health_url}", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        if _ready(health_url, 2.0) or _ready(models_url, 2.0):
            print("[vllm] Ready.", flush=True)
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for vLLM at {base_url}")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("[vllm] Stopping server...", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
    print("[vllm] Server stopped.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch vLLM, wait for its OpenAI API, then run a command.",
    )
    parser.add_argument("--model-path", default=os.environ.get("VLLM_MODEL_PATH", "~/XSkill/model/Qwen3.5-9B"))
    parser.add_argument("--model-name", default=os.environ.get("VLLM_MODEL_NAME", "Qwen3.5-9B"))
    parser.add_argument("--sif-path", default=os.environ.get("VLLM_SIF_PATH", ""))
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "18000")))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--tensor-parallel-size", type=int, default=_default_tp_size())
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
    )
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "0")) or None)
    parser.add_argument("--dtype", default=os.environ.get("VLLM_DTYPE", "bfloat16"))
    parser.add_argument("--reasoning-parser", default=os.environ.get("VLLM_REASONING_PARSER", "qwen3"))
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VLLM_ENFORCE_EAGER", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE", "1").lower() not in {"0", "false", "no"},
    )
    parser.add_argument("--tool-call-parser", default=os.environ.get("VLLM_TOOL_CALL_PARSER", "qwen3_xml"))
    parser.add_argument("--extra-args", default=os.environ.get("VLLM_EXTRA_ARGS", ""))
    parser.add_argument("--startup-timeout", type=int, default=int(os.environ.get("VLLM_STARTUP_TIMEOUT", "1800")))
    parser.add_argument(
        "--log-file",
        default=os.environ.get("VLLM_LOG_FILE", "benchmark/logs/vllm_qwen35_9b.log"),
        help="Path for vLLM stdout/stderr.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after vLLM is ready; prefix with --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to run after vLLM startup")
    return args


def main() -> int:
    args = parse_args()
    localhost_no_proxy = "127.0.0.1,localhost,0.0.0.0"
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        values = [item.strip() for item in current.split(",") if item.strip()]
        for item in localhost_no_proxy.split(","):
            if item not in values:
                values.append(item)
        os.environ[key] = ",".join(values)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "bond0")
    os.environ.setdefault("NCCL_IB_HCA", "mlx5_0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CHAT_TOOL_CALL_PARSER", args.tool_call_parser or "qwen3_xml")
    if "TRITON_CACHE_DIR" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("USER", "user")
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{job_id}"
    base_url = f"http://{args.host}:{args.port}"
    log_path = Path(args.log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    vllm_cmd = _build_vllm_cmd(args)
    print("[vllm] Command:", " ".join(shlex.quote(part) for part in vllm_cmd), flush=True)
    print(f"[vllm] Log file: {log_path}", flush=True)

    with open(log_path, "a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            vllm_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        try:
            _wait_for_vllm(proc, base_url, args.startup_timeout)
            env = os.environ.copy()
            env["no_proxy"] = os.environ["no_proxy"]
            env["NO_PROXY"] = os.environ["NO_PROXY"]
            env["BENCHMARK_BASE_URL"] = base_url.rstrip("/") + "/v1"
            env["BENCHMARK_API_KEY"] = args.api_key
            env["BENCHMARK_MODEL"] = args.model_name
            print("[run] Command:", " ".join(shlex.quote(part) for part in args.command), flush=True)
            print(f"[run] BENCHMARK_BASE_URL={env['BENCHMARK_BASE_URL']}", flush=True)
            print(f"[run] BENCHMARK_MODEL={env['BENCHMARK_MODEL']}", flush=True)
            completed = subprocess.run(args.command, env=env)
            return completed.returncode
        finally:
            _terminate_process_group(proc)


if __name__ == "__main__":
    raise SystemExit(main())
