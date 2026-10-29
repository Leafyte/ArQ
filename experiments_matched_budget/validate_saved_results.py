"""Read-only reconstruction and validator check of saved selections; never fits/selects/writes results."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from backend.domain import generate_routes  # noqa: E402
from backend.validator import validate_plan  # noqa: E402
import run_experiment as frozen  # noqa: E402

LOCKED = ["protocol.json", "frozen_lock.json", "dataset_manifest.json", "run_experiment.py", "results.json"]


def file_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest().upper() for name in LOCKED}


def main():
    before = file_hashes()
    lock = json.loads((HERE / "frozen_lock.json").read_text(encoding="utf-8"))
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert before["protocol.json"] == lock["protocol_sha256"]
    assert before["run_experiment.py"] == lock["runner_sha256"]
    assert frozen._sha(manifest) == lock["dataset_manifest_sha256"]
    records = frozen.build_records()
    rebuilt_manifest = frozen._manifest(records)
    assert rebuilt_manifest == manifest
    by_id = {record.scenario_id: record for record in records}

    method_counts = {}
    validated = 0
    objective_max_abs_diff = 0.0
    for method, rows in results["replicates"][0]["test_outcomes"].items():
        # Aggregate all five replicates from the persisted outcome records.
        all_rows = [row for rep in results["replicates"] for row in rep["test_outcomes"][method]]
        assert len(all_rows) == 30
        method_counts[method] = len(all_rows)
        for row in all_rows:
            record = by_id[row["scenario_id"]]
            assert record.split == "test"
            route_by_id = {frozen._route_id(route): route for route in generate_routes(record.realized)}
            assert len(set(row["selected_route_ids"])) == len(row["selected_route_ids"])
            assert all(route_id in route_by_id for route_id in row["selected_route_ids"])
            selected = sorted((route_by_id[key] for key in row["selected_route_ids"]),
                              key=lambda route: route["vehicle_id"])
            plan = {
                "status": "feasible",
                "routes": selected,
                "travel_cost": sum(float(route["travel_cost"]) for route in selected),
                "lateness_exposure": sum(float(route["lateness_exposure"]) for route in selected),
                "objective": sum(float(route["objective"]) for route in selected),
            }
            checked = validate_plan(record.realized, plan)
            assert checked["valid"], (method, row["scenario_id"], checked["errors"])
            actual = float(checked["recomputed"]["objective"])
            diff = abs(actual - float(row["validator_objective"]))
            objective_max_abs_diff = max(objective_max_abs_diff, diff)
            assert math.isclose(actual, float(row["validator_objective"]), rel_tol=1e-9, abs_tol=1e-9)
            validated += 1

    test_ids = {sid for rep in manifest["replicates"].values() for sid in rep["split_ids"]["test"]}
    assert len(test_ids) == 30
    for method in method_counts:
        ids = {row["scenario_id"] for rep in results["replicates"]
               for row in rep["test_outcomes"][method]}
        assert ids == test_ids
    after = file_hashes()
    assert after == before
    print(json.dumps({"locked_file_sha256_before": before, "locked_file_sha256_after": after,
                      "test_scenarios": 30, "methods": method_counts,
                      "selected_plan_validator_checks": validated,
                      "valid": validated == 90, "max_abs_objective_difference": objective_max_abs_diff,
                      "refit": False, "selection_recomputed": False,
                      "result_files_modified": False}, indent=2))


if __name__ == "__main__":
    main()
