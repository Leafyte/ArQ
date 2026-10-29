"""Checks the separately frozen matched-comparison design and input boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments_strong_baseline.run_experiment import (
    GroupedBootstrapCartForest, LOCK_PATH, MANIFEST_PATH, PROTOCOL_PATH,
    _decision_input, _manifest, build_records,
)


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def test_frozen_protocol_and_split_manifest_match_lock():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    protocol = PROTOCOL_PATH.read_bytes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    runner = (Path(__file__).parents[1] / "experiments_strong_baseline" / "run_experiment.py").read_bytes()
    assert hashlib.sha256(protocol).hexdigest().upper() == lock["protocol_sha256"]
    assert _canonical_hash(manifest) == lock["dataset_manifest_sha256"]
    assert hashlib.sha256(runner).hexdigest().upper() == lock["runner_sha256"]
    assert lock["locked_before_model_evaluation"] is True
    assert lock["no_heldout_tuning"] is True


def test_generated_scenario_groups_match_frozen_split_and_digest_manifest():
    records = build_records()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(records) == 150
    assert _manifest(records) == manifest
    for rep in range(1, 6):
        groups = [row for row in records if row.scenario_id.startswith(f"rep{rep:02d}-")]
        assert {split: sum(row.split == split for row in groups)
                for split in ("train", "validation", "test")} == {
                    "train": 18, "validation": 6, "test": 6}


def test_validation_and_test_labels_and_actual_train_plan_costs_are_hidden():
    records = build_records()
    for record in records:
        if record.split == "train":
            assert all(row.train_label in (0, 1) for row in record.routes)
            assert all(plan["realized_objective_train_only"] is not None for plan in record.plans)
        else:
            assert all(row.train_label is None for row in record.routes)
            assert all(plan["realized_objective_train_only"] is None for plan in record.plans)


def test_model_visible_decision_input_has_no_realized_scenario_or_outcomes():
    record = next(row for row in build_records() if row.split == "test")
    visible = _decision_input(record)
    assert not hasattr(visible, "realized")
    assert all(not hasattr(row, "train_label") for row in visible.routes)
    assert all("realized_objective_train_only" not in plan for plan in visible.plans)
    assert visible.forecast is record.forecast


def test_grouped_bootstrap_cart_forest_is_deterministic_and_nonlinear():
    groups = {}
    for group_index, (x0, x1, label) in enumerate(((0.1, 0.1, 0), (0.1, 0.9, 1),
                                                   (0.9, 0.1, 1), (0.9, 0.9, 0))):
        rows = []
        for i in range(16):
            dx = ((i * 7) % 13 - 6) * 0.002
            dy = ((i * 5) % 11 - 5) * 0.002
            rows.append(((x0 + dx, x1 + dy), label))
        groups[f"group-{group_index}"] = rows
    forest_a = GroupedBootstrapCartForest(n_estimators=64, max_depth=3,
                                         min_samples_leaf=2, seed=902).fit(groups)
    forest_b = GroupedBootstrapCartForest(n_estimators=64, max_depth=3,
                                         min_samples_leaf=2, seed=902).fit(groups)
    probes = ((0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9))
    predictions_a = [forest_a.predict_probability(row) for row in probes]
    predictions_b = [forest_b.predict_probability(row) for row in probes]
    assert predictions_a == predictions_b
    negative_mean = (predictions_a[0] + predictions_a[3]) / 2
    positive_mean = (predictions_a[1] + predictions_a[2]) / 2
    assert positive_mean - negative_mean > 0.1
    assert forest_a.node_count > forest_a.n_estimators

