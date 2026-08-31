"""Minimal reproducer for the SkyRL/vLLM prefix-cache reset contract bug.

Run against the pinned vLLM environment (no GPU or model is required)::

    /shared/ubuntu/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
      tests/backends/skyrl_train/inference_servers/repro_reset_prefix_cache_contract.py

The fake engine models vLLM's real busy-cache behavior: a reset without
``reset_running_requests`` fails, while a reset that preempts running requests
succeeds. The app registers vLLM's native route first and then the former SkyRL
duplicate, exactly matching ``vllm_server_actor._build_and_serve_vllm_server``.
"""

import asyncio
import json
from typing import Any

import httpx
from fastapi import FastAPI, Request
from vllm.entrypoints.serve.dev.cache.api_router import attach_router as attach_vllm_cache_router


class BusyEngineClient:
    """Small stand-in for a vLLM engine whose cache has active block holders."""

    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_external: bool = False,
    ) -> bool:
        self.calls.append(
            {
                "reset_running_requests": reset_running_requests,
                "reset_external": reset_external,
            }
        )
        return reset_running_requests


def build_app() -> tuple[FastAPI, BusyEngineClient, list[dict[str, Any]]]:
    app = FastAPI()
    engine = BusyEngineClient()
    app.state.engine_client = engine

    # vLLM's build_app() registers this native Query-based route first.
    attach_vllm_cache_router(app)

    duplicate_route_calls: list[dict[str, Any]] = []

    # This is the former SkyRL JSON-body route. Starlette dispatches the first
    # matching route, so this duplicate is never reached.
    @app.post("/reset_prefix_cache")
    async def former_skyrl_reset_prefix_cache(request: Request):
        body = await request.json()
        duplicate_route_calls.append(body)
        success = await engine.reset_prefix_cache(reset_running_requests=body.get("reset_running_requests", False))
        return {"success": success, "handler": "skyrl-duplicate"}

    return app, engine, duplicate_route_calls


async def main() -> None:
    app, engine, duplicate_route_calls = build_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://reproducer") as client:
        old_response = await client.post(
            "/reset_prefix_cache",
            json={"reset_running_requests": True},
        )
        fixed_response = await client.post(
            "/reset_prefix_cache",
            params={"reset_running_requests": "true", "reset_external": "true"},
        )

    matching_routes = [
        {
            "path": route.path,
            "handler": route.endpoint.__name__,
            "module": route.endpoint.__module__,
        }
        for route in app.routes
        if getattr(route, "path", None) == "/reset_prefix_cache"
    ]
    result = {
        "matching_routes_in_dispatch_order": matching_routes,
        "old_json_body_request": {
            "http_status": old_response.status_code,
            "response": old_response.json(),
        },
        "fixed_query_request": {
            "http_status": fixed_response.status_code,
            "response": fixed_response.json(),
        },
        "engine_calls": engine.calls,
        "skyrl_duplicate_route_calls": duplicate_route_calls,
    }

    assert len(matching_routes) == 2
    assert matching_routes[0]["module"] == "vllm.entrypoints.serve.dev.cache.api_router"
    assert old_response.status_code == 200
    assert old_response.json() == {"success": False}
    assert fixed_response.status_code == 200
    assert fixed_response.json() == {"success": True}
    assert duplicate_route_calls == []
    assert engine.calls == [
        {"reset_running_requests": False, "reset_external": False},
        {"reset_running_requests": True, "reset_external": True},
    ]

    print(json.dumps(result, indent=2, sort_keys=True))
    print("REPRODUCED: JSON flag ignored; query flag honored; duplicate SkyRL route unreachable")


if __name__ == "__main__":
    asyncio.run(main())
