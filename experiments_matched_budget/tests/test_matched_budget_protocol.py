import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def test_frozen_inputs_match_lock_and_runner_refuses_repeat():
    lock = json.loads((HERE / "frozen_lock.json").read_text(encoding="utf-8"))
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    manifest = json.loads((HERE / "dataset_manifest.json").read_text(encoding="utf-8"))
    runner_hash = hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest().upper()
    assert hashlib.sha256(protocol_bytes).hexdigest().upper() == lock["protocol_sha256"]
    assert canonical_sha(manifest) == lock["dataset_manifest_sha256"]
    assert runner_hash == lock["runner_sha256"]
    assert "if OUTPUT_PATH.exists():" in (HERE / "run_experiment.py").read_text(encoding="utf-8")


def test_protocol_has_independent_seeds_and_common_grouped_splits():
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((HERE / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert protocol["scenario_family"]["family_seed"] == 2026103001
    assert protocol["scenario_family"]["split_seed"] == 2026103002
    assert protocol["scenario_family"]["split_per_replicate"] == {"train": 18, "validation": 6, "test": 6}
    assert manifest["scenario_group_count"] == 150
    for rep in manifest["replicates"].values():
        assert {key: len(ids) for key, ids in rep["split_ids"].items()} == {"train": 18, "validation": 6, "test": 6}


def test_cpu_budget_and_work_counters_are_explicit():
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8-sig"))
    budgets = protocol["methods_and_fixed_budgets"]
    assert budgets["cpu_seconds_per_method_replicate"] == 0.25
    assert budgets["max_updates"] == 120
    source = (HERE / "run_experiment.py").read_text(encoding="utf-8")
    assert "process_time() - start_cpu < budget" in source
    assert '"decision_vqc_updates"' in source
    assert '"predictive_vqc_updates"' in source
    assert '"logistic_updates"' in source

def test_single_saved_evaluation_has_equal_scenario_counts_and_valid_plans():
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    methods = results["test_metrics_by_method"]
    assert set(methods) == {"decision_focused_vqc", "predictive_only_vqc", "logistic_regression"}
    assert all(row["n_scenarios"] == 30 for row in methods.values())
    assert all(row["valid_plan_rate"] == 1.0 for row in methods.values())
    assert all(len(results["per_scenario_paired_objective_differences_vs_logistic"][name]["per_scenario"]) == 30
               for name in ("decision_focused_vqc", "predictive_only_vqc"))
