#!/usr/bin/env python3
"""OpenAI-compatible request router for a fleet of one-GPU vLLM servers.

The router accepts non-streaming OpenAI-compatible requests, holds them for a
small batching window, then fans them out concurrently across backend vLLM
servers.  Each backend still performs its own internal vLLM batching; this
process makes sure bursts of benchmark requests are spread across all model
replicas instead of piling up behind a single server.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class Backend:
    name: str
    base_url: str
    in_flight: int = 0
    requests: int = 0
    failures: int = 0


@dataclass
class PendingRequest:
    request_id: str
    path: str
    body: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float


class BatchRouter:
    def __init__(
        self,
        *,
        backends: list[Backend],
        batch_window: float,
        max_batch_size: int,
        request_timeout: float,
        upstream_timeout: float,
        backend_api_key: str,
        router_api_key: str,
        force_model_name: str,
        max_queue_size: int,
        retry_attempts: int,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.request_timeout = request_timeout
        self.upstream_timeout = upstream_timeout
        self.backend_api_key = backend_api_key
        self.router_api_key = router_api_key
        self.force_model_name = force_model_name
        self.retry_attempts = retry_attempts
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._client: Any | None = None
        self._worker: asyncio.Task | None = None
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._cursor = 0

    async def start(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=self.upstream_timeout)
        self._worker = asyncio.create_task(self._batch_worker(), name="vllm-batch-router")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        for task in list(self._dispatch_tasks):
            task.cancel()
        if self._dispatch_tasks:
            await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)
        if self._client:
            await self._client.aclose()

    def check_auth(self, request: Request) -> None:
        from fastapi import HTTPException

        if not self.router_api_key:
            return
        header = request.headers.get("authorization", "")
        expected = f"Bearer {self.router_api_key}"
        if header != expected:
            raise HTTPException(status_code=401, detail="invalid or missing router API key")

    async def enqueue(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        from fastapi import HTTPException

        if body.get("stream"):
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported by the batching router; send non-streaming requests.",
            )
        if self.force_model_name:
            body = dict(body)
            body["model"] = self.force_model_name

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending = PendingRequest(
            request_id=uuid.uuid4().hex,
            path=path,
            body=body,
            future=future,
            enqueued_at=time.time(),
        )
        try:
            self.queue.put_nowait(pending)
        except asyncio.QueueFull as exc:
            raise HTTPException(status_code=503, detail="router queue is full") from exc

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise HTTPException(status_code=504, detail="request timed out in router") from exc

    async def backend_snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "name": backend.name,
                    "base_url": backend.base_url,
                    "in_flight": backend.in_flight,
                    "requests": backend.requests,
                    "failures": backend.failures,
                }
                for backend in self.backends
            ]

    async def _batch_worker(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_window

            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            print(
                f"[router] dispatch batch size={len(batch)} queued={self.queue.qsize()}",
                flush=True,
            )
            task = asyncio.create_task(self._dispatch_batch(batch))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch_batch(self, batch: list[PendingRequest]) -> None:
        await asyncio.gather(*(self._serve_request(item) for item in batch))

    async def _serve_request(self, pending: PendingRequest) -> None:
        last_error = ""
        attempts = max(1, self.retry_attempts + 1)
        for attempt in range(attempts):
            backend = await self._acquire_backend()
            try:
                status, payload = await self._post_to_backend(backend, pending)
                if status >= 500 and attempt + 1 < attempts:
                    last_error = f"backend {backend.name} returned HTTP {status}"
                    await self._mark_failure(backend)
                    continue
                if not pending.future.done():
                    pending.future.set_result((status, payload))
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._mark_failure(backend)
                if attempt + 1 >= attempts and not pending.future.done():
                    pending.future.set_result(
                        (
                            502,
                            {
                                "error": {
                                    "message": f"all backend attempts failed; last error: {last_error}",
                                    "type": "backend_error",
                                }
                            },
                        )
                    )
            finally:
                await self._release_backend(backend)

    async def _post_to_backend(self, backend: Backend, pending: PendingRequest) -> tuple[int, dict[str, Any]]:
        assert self._client is not None
        headers = {"Content-Type": "application/json", "X-Router-Request-Id": pending.request_id}
        if self.backend_api_key:
            headers["Authorization"] = f"Bearer {self.backend_api_key}"
        url = f"{backend.base_url.rstrip('/')}/{pending.path.lstrip('/')}"
        resp = await self._client.post(url, json=pending.body, headers=headers)
        try:
            payload = resp.json()
        except ValueError:
            payload = {
                "error": {
                    "message": resp.text,
                    "type": "non_json_backend_response",
                }
            }
        return resp.status_code, payload

    async def _acquire_backend(self) -> Backend:
        async with self._lock:
            min_in_flight = min(backend.in_flight for backend in self.backends)
            candidates = [backend for backend in self.backends if backend.in_flight == min_in_flight]
            backend = candidates[self._cursor % len(candidates)]
            self._cursor += 1
            backend.in_flight += 1
            backend.requests += 1
            return backend

    async def _release_backend(self, backend: Backend) -> None:
        async with self._lock:
            backend.in_flight = max(0, backend.in_flight - 1)

    async def _mark_failure(self, backend: Backend) -> None:
        async with self._lock:
            backend.failures += 1


def _parse_backends(values: list[str]) -> list[Backend]:
    backends: list[Backend] = []
    for i, value in enumerate(values):
        raw = value.strip()
        if not raw:
            continue
        if "=" in raw:
            name, base_url = raw.split("=", 1)
        else:
            name, base_url = f"vllm-{i}", raw
        backends.append(Backend(name=name.strip(), base_url=base_url.strip().rstrip("/")))
    return backends


def create_app(args: argparse.Namespace) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    # Endpoint annotations are postponed by ``from __future__ import annotations``.
    # Expose FastAPI's Request globally so FastAPI can resolve ``request: Request``
    # as the framework request object instead of treating it as user input.
    globals()["Request"] = Request

    router = BatchRouter(
        backends=_parse_backends(args.backend),
        batch_window=args.batch_window,
        max_batch_size=args.max_batch_size,
        request_timeout=args.request_timeout,
        upstream_timeout=args.upstream_timeout,
        backend_api_key=args.backend_api_key,
        router_api_key=args.router_api_key,
        force_model_name=args.force_model_name,
        max_queue_size=args.max_queue_size,
        retry_attempts=args.retry_attempts,
    )
    app = FastAPI(title="vLLM Batch Router")

    @app.on_event("startup")
    async def _startup() -> None:
        await router.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await router.stop()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "queued": router.queue.qsize(),
            "batch_window": router.batch_window,
            "max_batch_size": router.max_batch_size,
            "backends": await router.backend_snapshot(),
        }

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        router.check_auth(request)
        if router.force_model_name:
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": router.force_model_name,
                            "object": "model",
                            "owned_by": "vllm-batch-router",
                        }
                    ],
                }
            )
        assert router._client is not None
        backend = router.backends[0]
        headers = {}
        if router.backend_api_key:
            headers["Authorization"] = f"Bearer {router.backend_api_key}"
        try:
            resp = await router._client.get(f"{backend.base_url.rstrip('/')}/v1/models", headers=headers)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"backend models request failed: {exc}") from exc

    async def _handle_openai_request(path: str, request: Request) -> JSONResponse:
        router.check_auth(request)
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        status, payload = await router.enqueue(path, body)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await _handle_openai_request("/v1/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        return await _handle_openai_request("/v1/completions", request)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch and route OpenAI-compatible requests across vLLM backends.")
    parser.add_argument("--host", default=os.environ.get("VLLM_ROUTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_ROUTER_PORT", "19000")))
    parser.add_argument(
        "--backend",
        action="append",
        default=[item for item in os.environ.get("VLLM_BACKEND_URLS", "").split(",") if item],
        help="Backend root URL, optionally name=url. Repeat for each vLLM server.",
    )
    parser.add_argument("--batch-window", type=float, default=float(os.environ.get("VLLM_ROUTER_BATCH_WINDOW", "0.25")))
    parser.add_argument("--max-batch-size", type=int, default=int(os.environ.get("VLLM_ROUTER_MAX_BATCH_SIZE", "64")))
    parser.add_argument("--max-queue-size", type=int, default=int(os.environ.get("VLLM_ROUTER_MAX_QUEUE_SIZE", "4096")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("VLLM_ROUTER_REQUEST_TIMEOUT", "900")))
    parser.add_argument("--upstream-timeout", type=float, default=float(os.environ.get("VLLM_ROUTER_UPSTREAM_TIMEOUT", "900")))
    parser.add_argument("--retry-attempts", type=int, default=int(os.environ.get("VLLM_ROUTER_RETRY_ATTEMPTS", "1")))
    parser.add_argument("--backend-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--router-api-key", default=os.environ.get("VLLM_ROUTER_API_KEY", ""))
    parser.add_argument("--force-model-name", default=os.environ.get("VLLM_ROUTER_MODEL_NAME", ""))
    parser.add_argument("--log-level", default=os.environ.get("VLLM_ROUTER_LOG_LEVEL", "info"))
    args = parser.parse_args()
    if not args.backend:
        parser.error("at least one --backend or VLLM_BACKEND_URLS entry is required")
    if args.batch_window < 0:
        parser.error("--batch-window must be >= 0")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
