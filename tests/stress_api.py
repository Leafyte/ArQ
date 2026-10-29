"""Bounded API/job stress check; defaults to in-process handler/SQLite checks.

Run with `.venv\\Scripts\\python.exe tests\\stress_api.py`. Pass
`--base-url http://127.0.0.1:8000` from an environment with localhost sockets
enabled to add HTTP concurrent request/job checks. Never uses credentials.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import create_app
from backend.jobs import JobCapacityError, JobManager

def _scenario() -> dict[str, Any]:
    return json.loads((ROOT / "fixtures" / "dispatch_tiny.json").read_text())


def _route_call(app: Any, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
    for route in app.routes:
        if not isinstance(route, APIRoute) or method not in route.methods:
            continue
        if route.path != path:
            continue
        result = route.endpoint(payload) if route.body_field else route.endpoint()
        if isinstance(result, JSONResponse):
            return result.status_code, json.loads(result.body)
        return route.status_code or 200, result
    return 404, {"error": {"code": "not_found"}}


def _bounded_handler_and_job_stress() -> dict[str, Any]:
    import backend.jobs as jobs

    started = time.perf_counter()
    counts = {"invalid_payloads": 0, "concurrent_jobs_submitted": 0,
              "capacity_rejections": 0, "cancelled_jobs": 0,
              "timed_out_jobs": 0, "restart_recovered_jobs": 0,
              "secret_sentinels_redacted": 0}
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()
    original_solver = jobs.solve_exact

    def gated_solver(scenario):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            if active >= 2:
                both_started.set()
        release.wait(timeout=3)
        try:
            return original_solver(scenario)
        finally:
            with state_lock:
                active -= 1

    jobs.solve_exact = gated_solver
    try:
        with tempfile.TemporaryDirectory(prefix="dispatch-stress-", dir=ROOT / ".runtime") as temp:
            db = Path(temp) / "jobs.sqlite3"
            manager = JobManager(db, max_workers=2, timeout_seconds=2)
            app = create_app(manager)
            scenario = _scenario()
            invalid = dict(scenario, **{"private-sentinel-key": "DO_NOT_ECHO_SENTINEL"})
            status, body = _route_call(app, "POST", "/api/v1/scenarios/validate", invalid)
            counts["invalid_payloads"] += 1
            assert status == 422 and "DO_NOT_ECHO_SENTINEL" not in json.dumps(body)
            counts["secret_sentinels_redacted"] += 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
                submitted = list(callers.map(lambda _: manager.submit(scenario), range(2)))
            counts["concurrent_jobs_submitted"] = len(submitted)
            assert both_started.wait(timeout=2), "both bounded workers did not start"
            try:
                manager.submit(scenario)
            except JobCapacityError:
                counts["capacity_rejections"] += 1
            assert counts["capacity_rejections"] == 1
            assert manager.cancel(submitted[0]["id"])["status"] == "cancelled"
            counts["cancelled_jobs"] = 1
            # A terminal cancellation must not free a worker still solving.
            try:
                manager.submit(scenario)
            except JobCapacityError:
                counts["capacity_rejections"] += 1
            assert counts["capacity_rejections"] == 2
            release.set()
            for future in list(manager._futures.values()):
                future.result(timeout=3)
            assert manager.get(submitted[0]["id"])["status"] == "cancelled"
            assert manager.get(submitted[1]["id"])["status"] == "succeeded"
            manager.close(wait=True)

            timeout_db = Path(temp) / "timeout.sqlite3"
            timeout_manager = JobManager(timeout_db, max_workers=1, timeout_seconds=0.05)
            timeout_started = threading.Event()
            timeout_release = threading.Event()
            def slow_solver(value):
                timeout_started.set()
                timeout_release.wait(timeout=3)
                return original_solver(value)
            jobs.solve_exact = slow_solver
            timeout_job = timeout_manager.submit(scenario)
            assert timeout_started.wait(timeout=2)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and timeout_manager.get(timeout_job["id"])["status"] != "timed_out":
                threading.Event().wait(0.005)
            assert timeout_manager.get(timeout_job["id"])["status"] == "timed_out"
            counts["timed_out_jobs"] = 1
            try:
                timeout_manager.submit(scenario)
            except JobCapacityError:
                counts["capacity_rejections"] += 1
            assert counts["capacity_rejections"] == 3
            timeout_release.set()
            for future in list(timeout_manager._futures.values()):
                future.result(timeout=3)
            timeout_manager.close(wait=True)

            restart_db = Path(temp) / "restart.sqlite3"
            first = JobManager(restart_db, max_workers=1)
            first.close(wait=True)
            con = sqlite3.connect(restart_db)
            try:
                con.execute("""INSERT INTO jobs(id,status,scenario_json,submitted_at,deadline_epoch,
                             execution_type,solver) VALUES(?,?,?,?,?,?,?)""",
                            ("recovery-sentinel", "running", json.dumps(scenario), "now",
                             time.time() + 20, "local_classical", "exact_enumeration"))
                con.execute("INSERT INTO runtime_lock VALUES(1,?,?,?)",
                            (4294967295, "stale-owner", "stale"))
                con.commit()
            finally:
                con.close()
            recovered = JobManager(restart_db, max_workers=1)
            recovered_job = recovered.get("recovery-sentinel")
            assert recovered_job["status"] == "failed"
            assert recovered_job["error"]["code"] == "restart_interrupted"
            counts["restart_recovered_jobs"] = 1
            recovered.close(wait=True)
            del recovered, first, manager, timeout_manager, app
            gc.collect()
    finally:
        jobs.solve_exact = original_solver

    return {"mode": "direct_handler_and_sqlite_jobs", "counts": counts,
            "max_active_solver_threads": max_active, "worker_budget": 2,
            "max_orders": len(_scenario()["orders"]), "elapsed_seconds": round(time.perf_counter() - started, 3),
            "memory_measurement": "not captured", "external_calls": 0,
            "result": "PASS"}


def _http_stress(base_url: str) -> dict[str, Any]:
    scenario = _scenario()
    request_count = 0
    request_lock = threading.Lock()

    def request(method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        nonlocal request_count
        with request_lock:
            request_count += 1
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base_url.rstrip("/") + path, data=data, method=method,
                      headers={"content-type": "application/json"})
        try:
            with urlopen(req, timeout=3) as response:
                body = response.read()
                return response.status, json.loads(body) if body else {}
        except HTTPError as error:
            body = error.read()
            return error.code, json.loads(body) if body else {}

    started = time.perf_counter()
    health_status, health = request("GET", "/api/v1/health")
    assert health_status == 200 and health["status"] == "ok"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: request("POST", "/api/v1/scenarios/validate", scenario), range(8)))
    assert all(status == 200 and body["valid"] for status, body in results)
    invalid = dict(scenario, **{"secret-sentinel": "DO_NOT_ECHO_SENTINEL"})
    invalid_status, body = request("POST", "/api/v1/scenarios/validate", invalid)
    assert invalid_status == 422 and "DO_NOT_ECHO_SENTINEL" not in json.dumps(body)
    # Two overlapping jobs exercise the live worker cap and cancellation route.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda _: request("POST", "/api/v1/jobs", scenario), range(2)))
    assert all(status == 202 for status, _ in jobs)
    cancelled_status, cancelled = request("POST", f"/api/v1/jobs/{jobs[0][1]['id']}/cancel", {})
    assert cancelled_status == 200 and cancelled["status"] in {"cancelled", "succeeded"}
    deadline = time.monotonic() + 3
    second_job_id = jobs[1][1]["id"]
    while time.monotonic() < deadline:
        status, current = request("GET", f"/api/v1/jobs/{second_job_id}")
        if status == 200 and current["status"] in {"succeeded", "failed", "infeasible", "cancelled", "timed_out"}:
            break
        threading.Event().wait(0.02)
    assert current["status"] in {"succeeded", "failed", "infeasible", "cancelled", "timed_out"}
    return {"mode": "localhost_http", "requests": request_count, "concurrent_validation_requests": 4,
            "jobs_submitted": 2, "cancel_request_status": cancelled["status"],
            "polled_job_status": current["status"], "invalid_payload_status": invalid_status,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "result": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="optional running local API URL for HTTP smoke/load")
    args = parser.parse_args()
    result = _bounded_handler_and_job_stress()
    if args.base_url:
        result["http"] = _http_stress(args.base_url)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
