"""SQLite audit trail for validated static dispatch-recovery attempts.

Only canonical validated source inputs are stored. Each attempt is append-only;
failed attempts never replace the latest independently validated success.
The API must acquire the shared ``JobManager`` runtime lease before using this
store, so it follows the service's one-process-per-database contract.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecoveryAuditStore:
    """Durable, append-only recovery records in the backend SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _initialize(self) -> None:
        with self._connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS recovery_audits (
                audit_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('recovered','failed')),
                created_at TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                scope_sha256 TEXT NOT NULL,
                original_sha256 TEXT NOT NULL,
                updated_sha256 TEXT NOT NULL,
                original_plan_sha256 TEXT NOT NULL,
                locked_prefixes_sha256 TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                outcome_sha256 TEXT NOT NULL,
                original_scenario_json TEXT NOT NULL,
                updated_scenario_json TEXT NOT NULL,
                original_plan_json TEXT NOT NULL,
                locked_prefixes_json TEXT NOT NULL,
                change_penalty REAL NOT NULL,
                result_json TEXT,
                failure_json TEXT
            )""")
            columns = {row["name"] for row in con.execute("PRAGMA table_info(recovery_audits)")}
            if "outcome_sha256" not in columns:
                con.execute("ALTER TABLE recovery_audits ADD COLUMN outcome_sha256 TEXT")
            # Migrate prior local audit DBs without dropping any records.
            if "started_at" not in columns:
                con.execute("ALTER TABLE recovery_audits ADD COLUMN started_at TEXT")
                con.execute("UPDATE recovery_audits SET started_at=created_at WHERE started_at IS NULL")
            if "scope_sha256" not in columns:
                con.execute("ALTER TABLE recovery_audits ADD COLUMN scope_sha256 TEXT")
                rows = con.execute("SELECT audit_id,original_scenario_json,original_plan_json FROM recovery_audits").fetchall()
                for row in rows:
                    source = {"original_scenario": json.loads(row["original_scenario_json"]),
                              "original_plan": json.loads(row["original_plan_json"])}
                    con.execute("UPDATE recovery_audits SET scope_sha256=? WHERE audit_id=?",
                                (sha256_json(source), row["audit_id"]))
            con.execute("CREATE TABLE IF NOT EXISTS recovery_audit_success_by_scope (scope_sha256 TEXT PRIMARY KEY, latest_success_id TEXT NOT NULL)")
            rows = con.execute("SELECT audit_id,scope_sha256 FROM recovery_audits WHERE status='recovered' ORDER BY rowid").fetchall()
            for row in rows:
                con.execute("INSERT INTO recovery_audit_success_by_scope(scope_sha256,latest_success_id) VALUES(?,?) ON CONFLICT(scope_sha256) DO UPDATE SET latest_success_id=excluded.latest_success_id",
                            (row["scope_sha256"], row["audit_id"]))

    def append(self, *, status: str, original_scenario: dict[str, Any],
               updated_scenario: dict[str, Any], original_plan: dict[str, Any],
               locked_prefixes: dict[str, list[str]], change_penalty: float,
               result: dict[str, Any] | None = None,
               failure: dict[str, str] | None = None,
               started_at: str | None = None) -> dict[str, Any]:
        if status not in {"recovered", "failed"}:
            raise ValueError("invalid audit status")
        if (status == "recovered") != (result is not None) or (status == "failed") != (failure is not None):
            raise ValueError("audit result/failure does not match status")
        # JSON serialization is also a final allow_nan=False safety check.
        original_json = canonical_json(original_scenario)
        updated_json = canonical_json(updated_scenario)
        plan_json = canonical_json(original_plan)
        prefixes_json = canonical_json(locked_prefixes)
        result_json = canonical_json(result) if result is not None else None
        failure_json = canonical_json(failure) if failure is not None else None
        hashes = {
            "original_sha256": hashlib.sha256(original_json.encode()).hexdigest(),
            "updated_sha256": hashlib.sha256(updated_json.encode()).hexdigest(),
            "original_plan_sha256": hashlib.sha256(plan_json.encode()).hexdigest(),
            "locked_prefixes_sha256": hashlib.sha256(prefixes_json.encode()).hexdigest(),
        }
        request_bundle = {"original_scenario": original_scenario,
                          "updated_scenario": updated_scenario,
                          "original_plan": original_plan,
                          "locked_prefixes": locked_prefixes,
                          "change_penalty": change_penalty}
        hashes["request_sha256"] = sha256_json(request_bundle)
        scope_sha256 = sha256_json({"original_scenario": original_scenario,
                                    "original_plan": original_plan})
        outcome_json = result_json if result_json is not None else failure_json
        hashes["outcome_sha256"] = hashlib.sha256(outcome_json.encode()).hexdigest()
        audit_id, completed_at = str(uuid.uuid4()), _now()
        started_at = started_at or completed_at
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("""INSERT INTO recovery_audits (
                audit_id,status,created_at,completed_at,original_sha256,updated_sha256,
                original_plan_sha256,locked_prefixes_sha256,request_sha256,outcome_sha256,
                original_scenario_json,updated_scenario_json,original_plan_json,
                locked_prefixes_json,change_penalty,result_json,failure_json,started_at,scope_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (audit_id, status, started_at, completed_at, hashes["original_sha256"],
                 hashes["updated_sha256"], hashes["original_plan_sha256"],
                 hashes["locked_prefixes_sha256"], hashes["request_sha256"], hashes["outcome_sha256"],
                 original_json, updated_json, plan_json, prefixes_json,
                 float(change_penalty), result_json, failure_json, started_at, scope_sha256))
            if status == "recovered":
                con.execute("INSERT INTO recovery_audit_success_by_scope(scope_sha256,latest_success_id) VALUES(?,?) ON CONFLICT(scope_sha256) DO UPDATE SET latest_success_id=excluded.latest_success_id", (scope_sha256, audit_id))
        return self.get(audit_id)  # type: ignore[return-value]

    def get(self, audit_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM recovery_audits WHERE audit_id=?", (audit_id,)).fetchone()
        return self._decode(row) if row else None

    def latest(self, scope_sha256: str | None = None) -> dict[str, Any]:
        with self._connect() as con:
            attempt = (con.execute("SELECT * FROM recovery_audits WHERE scope_sha256=? ORDER BY rowid DESC LIMIT 1", (scope_sha256,)).fetchone()
                       if scope_sha256 else con.execute("SELECT * FROM recovery_audits ORDER BY rowid DESC LIMIT 1").fetchone())
            selected_scope = scope_sha256 or (attempt["scope_sha256"] if attempt else None)
            state = con.execute("SELECT latest_success_id FROM recovery_audit_success_by_scope WHERE scope_sha256=?", (selected_scope,)).fetchone() if selected_scope else None
            success = None
            if state and state["latest_success_id"]:
                success = con.execute("SELECT * FROM recovery_audits WHERE audit_id=?",
                                      (state["latest_success_id"],)).fetchone()
        return {"latest_attempt": self._decode(attempt) if attempt else None,
                "latest_success": self._decode(success) if success else None}

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "audit_id": row["audit_id"], "status": row["status"],
            "started_at": row["started_at"], "completed_at": row["completed_at"],
            "scope_sha256": row["scope_sha256"],
            "hashes": {"original_sha256": row["original_sha256"],
                       "updated_sha256": row["updated_sha256"],
                       "original_plan_sha256": row["original_plan_sha256"],
                       "locked_prefixes_sha256": row["locked_prefixes_sha256"],
                       "request_sha256": row["request_sha256"],
                       "outcome_sha256": row["outcome_sha256"]},
            "original_scenario": json.loads(row["original_scenario_json"]),
            "updated_scenario": json.loads(row["updated_scenario_json"]),
            "original_plan": json.loads(row["original_plan_json"]),
            "locked_prefixes": json.loads(row["locked_prefixes_json"]),
            "change_penalty": row["change_penalty"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "failure": json.loads(row["failure_json"]) if row["failure_json"] else None,
        }
