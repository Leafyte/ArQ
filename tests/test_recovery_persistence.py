import json
import sqlite3
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from backend.api import create_app
from backend.domain import solve_exact
from backend.jobs import JobManager


class Response:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body.decode("utf-8")

    def json(self):
        return json.loads(self.text)


class DirectRoutes:
    """Exercise registered route handlers without blocked loopback sockets."""
    def __init__(self, app):
        self.app = app

    def call(self, method, path, payload=None):
        parsed = urlsplit(path)
        path = parsed.path
        query = parse_qs(parsed.query)
        for route in self.app.routes:
            if not isinstance(route, APIRoute) or method not in route.methods:
                continue
            if route.path == path:
                value = route.endpoint(payload) if route.body_field else route.endpoint(
                    scope_sha256=query.get("scope_sha256", [None])[0]) if "latest" in path else route.endpoint()
            elif "{audit_id}" in route.path:
                prefix, suffix = route.path.split("{audit_id}")
                if not path.startswith(prefix) or not path.endswith(suffix):
                    continue
                value = route.endpoint(path[len(prefix):len(path) - len(suffix) if suffix else None])
            else:
                continue
            if isinstance(value, JSONResponse):
                return Response(value.status_code, value.body)
            return Response(route.status_code or 200, json.dumps(value).encode())
        return Response(404, b'{"error":{"code":"not_found"}}')


@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())


@pytest.fixture
def runtime(tmp_path):
    manager = JobManager(tmp_path / "audit.sqlite3", max_workers=1)
    client = DirectRoutes(create_app(manager))
    yield manager, client
    manager.close(wait=True)


def _valid_request(scenario):
    scenario = json.loads(json.dumps(scenario))
    plan = solve_exact(scenario)
    prefixes = {route["vehicle_id"]: route["order_ids"][:1] for route in plan["routes"]}
    return {"original_scenario": scenario, "original_plan": plan,
            "updated_scenario": json.loads(json.dumps(scenario)),
            "locked_prefixes": prefixes, "change_penalty": 0.5}


def test_recovery_audit_persists_validated_inputs_hashes_and_result_across_store_restart(runtime, scenario):
    manager, client = runtime
    response = client.call("POST", "/api/v1/recovery/plan", _valid_request(scenario))
    assert response.status_code == 200
    result = response.json()
    audit_id = result["audit"]["audit_id"]
    assert result["validation"]["valid"]
    assert all(len(value) == 64 for value in result["audit"]["hashes"].values())
    assert result["audit"]["started_at"] <= result["audit"]["completed_at"]

    # Release and reacquire the runtime lease to simulate backend restart.
    db_path = manager.db_path
    manager.close(wait=True)
    restarted_manager = JobManager(db_path, max_workers=1)
    restarted_client = DirectRoutes(create_app(restarted_manager))
    record = restarted_client.call("GET", f"/api/v1/recovery/audits/{audit_id}").json()
    assert record["status"] == "recovered"
    assert record["result"]["validation"]["valid"]
    assert record["original_plan"] == result["original_plan"]
    assert record["locked_prefixes"]
    latest = restarted_client.call("GET", "/api/v1/recovery/audits/latest").json()
    assert latest["latest_success"]["audit_id"] == audit_id
    assert latest["latest_attempt"]["audit_id"] == audit_id
    restarted_manager.close(wait=True)


def test_failed_recovery_is_recorded_without_replacing_latest_valid_plan(runtime, scenario):
    manager, client = runtime
    success = client.call("POST", "/api/v1/recovery/plan", _valid_request(scenario)).json()
    successful_audit_id = success["audit"]["audit_id"]

    failed_request = _valid_request(scenario)
    locked_vehicle = next(iter(failed_request["locked_prefixes"]))
    failed_request["updated_scenario"]["vehicles"] = [
        vehicle for vehicle in failed_request["updated_scenario"]["vehicles"]
        if vehicle["id"] != locked_vehicle]
    failed = client.call("POST", "/api/v1/recovery/plan", failed_request)
    assert failed.status_code == 409
    error = failed.json()["error"]
    assert error["code"] == "locked_work_vehicle_unavailable"
    failed_audit_id = error["audit"]["audit_id"]

    latest = client.call("GET", "/api/v1/recovery/audits/latest").json()
    assert latest["latest_attempt"]["audit_id"] == failed_audit_id
    assert latest["latest_attempt"]["status"] == "failed"
    assert latest["latest_attempt"]["failure"]["code"] == error["code"]
    assert latest["latest_success"]["audit_id"] == successful_audit_id
    assert latest["latest_success"]["result"]["validation"]["valid"]

    with sqlite3.connect(manager.db_path) as con:
        rows = con.execute("SELECT status FROM recovery_audits ORDER BY rowid").fetchall()
    assert [row[0] for row in rows] == ["recovered", "failed"]


def test_latest_success_is_scoped_to_original_scenario_and_plan(runtime, scenario):
    _, client = runtime
    first = _valid_request(scenario)
    first_success = client.call("POST", "/api/v1/recovery/plan", first).json()
    second = _valid_request(scenario)
    second["original_scenario"]["vehicles"][0]["capacity"] += 1
    second["updated_scenario"]["vehicles"][0]["capacity"] += 1
    # The original route remains independently valid under the changed capacity.
    second_success = client.call("POST", "/api/v1/recovery/plan", second)
    assert second_success.status_code == 200
    second_success = second_success.json()
    assert first_success["audit"]["scope_sha256"] != second_success["audit"]["scope_sha256"]

    failed = _valid_request(scenario)
    locked_vehicle = next(iter(failed["locked_prefixes"]))
    failed["updated_scenario"]["vehicles"] = [
        vehicle for vehicle in failed["updated_scenario"]["vehicles"]
        if vehicle["id"] != locked_vehicle]
    failed_response = client.call("POST", "/api/v1/recovery/plan", failed)
    assert failed_response.status_code == 409
    # Default latest view follows latest attempt's scope, while explicit lookup
    # returns the second identity's own successful plan.
    first_latest = client.call("GET", "/api/v1/recovery/audits/latest").json()
    assert first_latest["latest_attempt"]["status"] == "failed"
    assert first_latest["latest_success"]["scope_sha256"] == first_latest["latest_attempt"]["scope_sha256"]
    assert first_latest["latest_success"]["audit_id"] != second_success["audit"]["audit_id"]
    second_latest = client.call("GET", "/api/v1/recovery/audits/latest?scope_sha256=" +
                                second_success["audit"]["scope_sha256"]).json()
    assert second_latest["latest_attempt"]["audit_id"] == second_success["audit"]["audit_id"]
    assert second_latest["latest_success"]["audit_id"] == second_success["audit"]["audit_id"]


def test_invalid_and_secret_bearing_fields_are_rejected_before_audit_storage(runtime, scenario):
    manager, client = runtime
    for location in ("scenario", "plan"):
        payload = _valid_request(scenario)
        if location == "scenario":
            payload["original_scenario"]["request_secret"] = "DO_NOT_STORE_SENTINEL"
        else:
            payload["original_plan"]["secret_token"] = "DO_NOT_STORE_SENTINEL"
        response = client.call("POST", "/api/v1/recovery/plan", payload)
        assert response.status_code == 422
        assert "DO_NOT_STORE_SENTINEL" not in response.text
    with sqlite3.connect(manager.db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='recovery_audits'").fetchone()[0] == 0


def test_read_only_audit_lookup_has_stable_not_found(runtime):
    _, client = runtime
    assert client.call("GET", "/api/v1/recovery/audits/latest").json() == {
        "latest_attempt": None, "latest_success": None}
    missing = client.call("GET", "/api/v1/recovery/audits/not-an-audit")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "audit_not_found"
