#!/usr/bin/env python3
"""Start a persistent vLLM fleet and batching router inside a Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import launch_vllm_fleet_then_run as fleet


def _make_api_key() -> str:
    return secrets.token_urlsafe(32)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_has_live_process(state_path: Path) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for item in state.get("processes", []):
        pid = int(item.get("pid") or 0)
        if pid > 0 and _pid_is_running(pid):
            return True
    return False


def _advertise_host(router_host: str) -> str:
    if router_host not in {"", "0.0.0.0", "::"}:
        return router_host
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        for item in output.split():
            if item and not item.startswith("127."):
                return item
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return socket.gethostname()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def _default_state_file(logs_dir: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID") or "manual"
    return logs_dir / f"vllm_fleet_state_{job_id}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start one vLLM replica per GPU plus an OpenAI-compatible batching router.",
    )
    parser.add_argument("--model-path", default=os.environ.get("VLLM_MODEL_PATH", "~/XSkill/model/Qwen3.5-9B"))
    parser.add_argument("--model-name", default=os.environ.get("VLLM_MODEL_NAME", "Qwen3.5-9B"))
    parser.add_argument("--sif-path", default=os.environ.get("VLLM_SIF_PATH", ""))
    parser.add_argument("--vllm-host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("VLLM_BASE_PORT", "18000")))
    parser.add_argument("--router-host", default=os.environ.get("VLLM_ROUTER_HOST", "0.0.0.0"))
    parser.add_argument("--router-port", type=int, default=int(os.environ.get("VLLM_ROUTER_PORT", "19000")))
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("VLLM_BACKEND_API_KEY") or os.environ.get("VLLM_API_KEY", ""),
        help="API key required by each backend vLLM replica. Generated when omitted.",
    )
    parser.add_argument(
        "--router-api-key",
        default=os.environ.get("VLLM_ROUTER_API_KEY", ""),
        help="API key required by clients calling the batching router. Generated when omitted.",
    )
    parser.add_argument("--num-servers", type=int, default=int(os.environ.get("VLLM_NUM_SERVERS", "8")))
    parser.add_argument("--devices", default=os.environ.get("VLLM_CUDA_DEVICES", ""))
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
    parser.add_argument("--logs-dir", default=os.environ.get("VLLM_LOGS_DIR", "benchmark/logs/vllm_fleet"))
    parser.add_argument("--state-file", default=os.environ.get("VLLM_FLEET_STATE_FILE", ""))
    parser.add_argument("--router-batch-window", type=float, default=float(os.environ.get("VLLM_ROUTER_BATCH_WINDOW", "0.25")))
    parser.add_argument("--router-max-batch-size", type=int, default=int(os.environ.get("VLLM_ROUTER_MAX_BATCH_SIZE", "64")))
    parser.add_argument("--router-request-timeout", type=float, default=float(os.environ.get("VLLM_ROUTER_REQUEST_TIMEOUT", "900")))
    parser.add_argument("--force", action="store_true", help="Overwrite a stale state file when no recorded PID is live.")
    args = parser.parse_args()
    if args.num_servers < 1:
        parser.error("--num-servers must be >= 1")
    if not args.backend_api_key or args.backend_api_key == "EMPTY":
        args.backend_api_key = _make_api_key()
    if not args.router_api_key or args.router_api_key == "EMPTY":
        args.router_api_key = _make_api_key()
    return args


def main() -> int:
    args = parse_args()
    fleet._localhost_no_proxy()
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "bond0")
    os.environ.setdefault("NCCL_IB_HCA", "mlx5_0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CHAT_TOOL_CALL_PARSER", args.tool_call_parser or "qwen3_xml")

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else _default_state_file(logs_dir)
    if state_path.exists() and _state_has_live_process(state_path):
        print(f"[start] ERROR: live vLLM fleet already recorded in {state_path}", file=sys.stderr)
        print("[start] Run benchmark/scripts/stop_qwen35_9b_vllm_fleet.sh first.", file=sys.stderr)
        return 2
    if state_path.exists() and not args.force:
        print(f"[start] ERROR: state file exists: {state_path}", file=sys.stderr)
        print("[start] Use --force only after confirming the recorded processes are gone.", file=sys.stderr)
        return 2

    devices = fleet._devices(args.num_servers, args.devices)
    logs_dir.mkdir(parents=True, exist_ok=True)
    vllm_procs: list[subprocess.Popen] = []
    router_proc: subprocess.Popen | None = None
    try:
        backend_urls: list[str] = []
        health_urls: list[str] = []
        for idx, device in enumerate(devices):
            port = args.base_port + idx
            backend_urls.append(f"http://{args.vllm_host}:{port}")
            health_urls.append(f"http://127.0.0.1:{port}/health")
            cmd = fleet._build_vllm_cmd(args, port)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["VLLM_API_KEY"] = args.backend_api_key
            job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("USER", "manual")
            env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{job_id}_{idx}"
            log_path = logs_dir / f"qwen35_9b_replica_{idx}_{job_id}.log"
            print(f"[vllm-{idx}] device={device} port={port} log={log_path}", flush=True)
            print(f"[vllm-{idx}] Command: {fleet._redacted_command(cmd)}", flush=True)
            with open(log_path, "a", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            vllm_procs.append(proc)

        fleet._wait_for_urls(vllm_procs, health_urls, args.startup_timeout, "vllm")

        router_script = Path(__file__).resolve().parent / "vllm_batch_router.py"
        job_id = os.environ.get("SLURM_JOB_ID") or "manual"
        router_log = logs_dir / f"router_{job_id}.log"
        router_cmd = [
            sys.executable,
            str(router_script),
            "--host",
            args.router_host,
            "--port",
            str(args.router_port),
            "--force-model-name",
            args.model_name,
            "--batch-window",
            str(args.router_batch_window),
            "--max-batch-size",
            str(args.router_max_batch_size),
            "--request-timeout",
            str(args.router_request_timeout),
        ]
        for idx, url in enumerate(backend_urls):
            router_cmd += ["--backend", f"vllm-{idx}={url}"]

        print("[router] Command:", fleet._redacted_command(router_cmd), flush=True)
        print(f"[router] Log file: {router_log}", flush=True)
        router_env = os.environ.copy()
        router_env["VLLM_API_KEY"] = args.backend_api_key
        router_env["VLLM_ROUTER_API_KEY"] = args.router_api_key
        with open(router_log, "a", encoding="utf-8") as log_f:
            router_proc = subprocess.Popen(
                router_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=router_env,
            )
        fleet._wait_for_urls([router_proc], [f"http://127.0.0.1:{args.router_port}/health"], 120, "router")

        advertise_host = os.environ.get("VLLM_ADVERTISE_HOST", "") or _advertise_host(args.router_host)
        router_url = f"http://{advertise_host}:{args.router_port}/v1"
        state = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "advertise_host": advertise_host,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "model_name": args.model_name,
            "router_url": router_url,
            "router_api_key": args.router_api_key,
            "backend_api_key": args.backend_api_key,
            "logs_dir": str(logs_dir),
            "processes": [
                {
                    "label": "router",
                    "pid": router_proc.pid,
                    "pgid": os.getpgid(router_proc.pid),
                    "log": str(router_log),
                },
                *[
                    {
                        "label": f"vllm-{idx}",
                        "pid": proc.pid,
                        "pgid": os.getpgid(proc.pid),
                        "url": backend_urls[idx],
                        "log": str(logs_dir / f"qwen35_9b_replica_{idx}_{job_id}.log"),
                    }
                    for idx, proc in enumerate(vllm_procs)
                ],
            ],
        }
        _write_state(state_path, state)
        print(f"[state] {state_path}", flush=True)
        print("[client] export BENCHMARK_BASE_URL=%s" % shlex.quote(router_url), flush=True)
        print("[client] export BENCHMARK_API_KEY=%s" % shlex.quote(args.router_api_key), flush=True)
        print("[client] export BENCHMARK_MODEL=%s" % shlex.quote(args.model_name), flush=True)
        print("[start] vLLM fleet is ready; keep the Slurm allocation alive until stop.", flush=True)
        return 0
    except Exception:
        if router_proc is not None:
            fleet._terminate_process_group(router_proc, "router")
        for idx, proc in enumerate(vllm_procs):
            fleet._terminate_process_group(proc, f"vllm-{idx}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
