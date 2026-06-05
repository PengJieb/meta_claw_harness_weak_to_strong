#!/usr/bin/env python3
"""Launch one vLLM replica per GPU, route requests through a batch proxy, run a command."""

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


def _localhost_no_proxy() -> None:
    localhost = "127.0.0.1,localhost,0.0.0.0"
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        values = [item.strip() for item in current.split(",") if item.strip()]
        for item in localhost.split(","):
            if item not in values:
                values.append(item)
        os.environ[key] = ",".join(values)


def _devices(num_servers: int, devices_arg: str) -> list[str]:
    if devices_arg:
        devices = [item.strip() for item in devices_arg.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(i) for i in range(num_servers)]
    if len(devices) < num_servers:
        raise ValueError(f"need {num_servers} CUDA devices, got {len(devices)}: {devices}")
    return devices[:num_servers]


def _build_vllm_cmd(args: argparse.Namespace, port: int) -> list[str]:
    server_cmd = [
        "python3" if args.sif_path else sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    model_path = _expand(args.model_path)
    server_cmd += [
        "--model",
        "/model_weights" if args.sif_path else model_path,
        "--served-model-name",
        args.model_name,
        "--host",
        args.vllm_host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len:
        server_cmd += ["--max-model-len", str(args.max_model_len)]
    if args.dtype:
        server_cmd += ["--dtype", args.dtype]
    if args.trust_remote_code:
        server_cmd.append("--trust-remote-code")
    if args.backend_api_key:
        server_cmd += ["--api-key", args.backend_api_key]
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


def _redacted_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    for part in cmd:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(part)
        if part in {"--api-key", "--backend-api-key", "--router-api-key"}:
            redact_next = True
    return " ".join(shlex.quote(part) for part in redacted)


def _ready(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _wait_for_urls(procs: list[subprocess.Popen], urls: list[str], timeout_s: int, label: str) -> None:
    ready = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for idx, proc in enumerate(procs):
            if proc.poll() is not None:
                raise RuntimeError(f"{label} process {idx} exited early with code {proc.returncode}")
        for idx, url in enumerate(urls):
            if idx in ready:
                continue
            if _ready(url, 2.0):
                ready.add(idx)
                print(f"[{label}] ready {idx + 1}/{len(urls)}: {url}", flush=True)
        if len(ready) == len(urls):
            return
        time.sleep(5)
    missing = [url for idx, url in enumerate(urls) if idx not in ready]
    raise TimeoutError(f"timed out waiting for {label}; missing={missing}")


def _terminate_process_group(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    print(f"[cleanup] stopping {label} pid={proc.pid}", flush=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a vLLM replica fleet, batch-router, then run a command.")
    parser.add_argument("--model-path", default=os.environ.get("VLLM_MODEL_PATH", "~/XSkill/model/Qwen3.5-9B"))
    parser.add_argument("--model-name", default=os.environ.get("VLLM_MODEL_NAME", "Qwen3.5-9B"))
    parser.add_argument("--sif-path", default=os.environ.get("VLLM_SIF_PATH", ""))
    parser.add_argument("--vllm-host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("VLLM_BASE_PORT", "18000")))
    parser.add_argument("--router-host", default=os.environ.get("VLLM_ROUTER_HOST", "127.0.0.1"))
    parser.add_argument("--router-port", type=int, default=int(os.environ.get("VLLM_ROUTER_PORT", "19000")))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("VLLM_BACKEND_API_KEY") or os.environ.get("VLLM_API_KEY", ""),
        help="API key required by each backend vLLM replica.",
    )
    parser.add_argument(
        "--router-api-key",
        default=os.environ.get("VLLM_ROUTER_API_KEY", ""),
        help="API key required by the batching router client endpoint.",
    )
    parser.add_argument("--num-servers", type=int, default=int(os.environ.get("VLLM_NUM_SERVERS", "8")))
    parser.add_argument("--devices", default=os.environ.get("VLLM_CUDA_DEVICES", ""))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")))
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
    parser.add_argument("--logs-dir", default=os.environ.get("VLLM_LOGS_DIR", "benchmark/logs/vllm"))
    parser.add_argument("--router-batch-window", type=float, default=float(os.environ.get("VLLM_ROUTER_BATCH_WINDOW", "0.25")))
    parser.add_argument("--router-max-batch-size", type=int, default=int(os.environ.get("VLLM_ROUTER_MAX_BATCH_SIZE", "64")))
    parser.add_argument("--router-request-timeout", type=float, default=float(os.environ.get("VLLM_ROUTER_REQUEST_TIMEOUT", "900")))
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after startup; prefix with --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to run after vLLM fleet startup")
    if args.num_servers < 1:
        parser.error("--num-servers must be >= 1")
    if not args.backend_api_key:
        args.backend_api_key = args.api_key
    if not args.backend_api_key or args.backend_api_key == "EMPTY":
        parser.error("--backend-api-key or VLLM_BACKEND_API_KEY must be set for the vLLM replicas")
    if not args.router_api_key or args.router_api_key == "EMPTY":
        parser.error("--router-api-key or VLLM_ROUTER_API_KEY must be set for the wrapper")
    return args


def main() -> int:
    args = parse_args()
    _localhost_no_proxy()
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "bond0")
    os.environ.setdefault("NCCL_IB_HCA", "mlx5_0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("CHAT_TOOL_CALL_PARSER", args.tool_call_parser or "qwen3_xml")

    devices = _devices(args.num_servers, args.devices)
    logs_dir = Path(args.logs_dir).expanduser()
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
            cmd = _build_vllm_cmd(args, port)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("USER", "manual")
            env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{job_id}_{idx}"
            log_path = logs_dir / f"qwen35_9b_replica_{idx}_{os.environ.get('SLURM_JOB_ID', 'manual')}.log"
            print(
                f"[vllm-{idx}] device={device} port={port} log={log_path}",
                flush=True,
            )
            print("[vllm-%d] Command: %s" % (idx, _redacted_command(cmd)), flush=True)
            with open(log_path, "a", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            vllm_procs.append(proc)

        _wait_for_urls(vllm_procs, health_urls, args.startup_timeout, "vllm")

        router_script = Path(__file__).resolve().parent / "vllm_batch_router.py"
        router_log = logs_dir / f"router_{os.environ.get('SLURM_JOB_ID', 'manual')}.log"
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

        print("[router] Command:", _redacted_command(router_cmd), flush=True)
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
        _wait_for_urls([router_proc], [f"http://127.0.0.1:{args.router_port}/health"], 120, "router")

        env = os.environ.copy()
        env["no_proxy"] = os.environ["no_proxy"]
        env["NO_PROXY"] = os.environ["NO_PROXY"]
        env["BENCHMARK_BASE_URL"] = f"http://{args.router_host}:{args.router_port}/v1"
        env["BENCHMARK_API_KEY"] = args.router_api_key
        env["BENCHMARK_MODEL"] = args.model_name
        print("[run] Command:", " ".join(shlex.quote(part) for part in args.command), flush=True)
        print(f"[run] BENCHMARK_BASE_URL={env['BENCHMARK_BASE_URL']}", flush=True)
        completed = subprocess.run(args.command, env=env)
        return completed.returncode
    finally:
        if router_proc is not None:
            _terminate_process_group(router_proc, "router")
        for idx, proc in enumerate(vllm_procs):
            _terminate_process_group(proc, f"vllm-{idx}")


if __name__ == "__main__":
    raise SystemExit(main())
