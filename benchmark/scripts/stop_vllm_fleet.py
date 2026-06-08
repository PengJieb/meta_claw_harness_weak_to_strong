#!/usr/bin/env python3
"""Stop a persistent vLLM fleet started by start_vllm_fleet.py."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: signal.Signals, label: str) -> None:
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, sig)
        print(f"[stop] sent {sig.name} to {label} pgid={pgid}", flush=True)
    except ProcessLookupError:
        print(f"[stop] {label} pgid={pgid} is already gone", flush=True)
    except PermissionError:
        print(f"[stop] no permission to signal {label} pgid={pgid}", flush=True)


def _wait_gone(processes: list[dict[str, Any]], timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        live = [item for item in processes if _pid_is_running(int(item.get("pid") or 0))]
        if not live:
            return True
        time.sleep(1)
    return False


def _default_state_file(logs_dir: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID") or "manual"
    return logs_dir / f"vllm_fleet_state_{job_id}.json"


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stop_state_file(path: Path, *, keep_state: bool, timeout_s: float) -> int:
    if not path.exists():
        print(f"[stop] state file not found: {path}")
        return 1
    state = _load_state(path)
    processes = list(state.get("processes", []))
    if not processes:
        print(f"[stop] no processes recorded in {path}")
        if not keep_state:
            path.unlink(missing_ok=True)
        return 0

    print(f"[stop] state={path}")
    print(f"[stop] router_url={state.get('router_url', '')}")
    for item in processes:
        _signal_group(int(item.get("pgid") or 0), signal.SIGTERM, str(item.get("label") or item.get("pid")))
    if not _wait_gone(processes, timeout_s):
        for item in processes:
            if _pid_is_running(int(item.get("pid") or 0)):
                _signal_group(int(item.get("pgid") or 0), signal.SIGKILL, str(item.get("label") or item.get("pid")))
        _wait_gone(processes, 10)

    live = [item for item in processes if _pid_is_running(int(item.get("pid") or 0))]
    if live:
        print("[stop] WARNING: some recorded processes are still live:")
        for item in live:
            print(f"  {item.get('label')} pid={item.get('pid')} pgid={item.get('pgid')}")
        return 2

    if not keep_state:
        path.unlink(missing_ok=True)
        print(f"[stop] removed state file: {path}")
    print("[stop] vLLM fleet stopped")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop a vLLM fleet started by start_vllm_fleet.py.")
    parser.add_argument("--logs-dir", default=os.environ.get("VLLM_LOGS_DIR", "benchmark/logs/vllm_fleet"))
    parser.add_argument("--state-file", default=os.environ.get("VLLM_FLEET_STATE_FILE", ""))
    parser.add_argument("--all", action="store_true", help="Stop every vllm_fleet_state_*.json file in logs-dir.")
    parser.add_argument("--keep-state", action="store_true", help="Do not delete the state file after stopping.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).expanduser().resolve()
    if args.all:
        paths = sorted(logs_dir.glob("vllm_fleet_state_*.json"))
        if not paths:
            print(f"[stop] no state files found under {logs_dir}")
            return 1
        rc = 0
        for path in paths:
            rc = max(rc, stop_state_file(path, keep_state=args.keep_state, timeout_s=args.timeout))
        return rc
    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else _default_state_file(logs_dir)
    return stop_state_file(state_path, keep_state=args.keep_state, timeout_s=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
