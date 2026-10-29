import unittest
import json
from pathlib import Path

from backend.quantum_learning import (
    _make_examples,
    decision_loss_gradient,
    expectation_z0,
    finite_difference_gradient,
    risk_probability,
    run_study,
)
from backend.learning_dispatch import (
    _decision_objective_and_gradient,
    build_scenarios,
    finite_difference_decision_gradient,
    run_dispatch_study,
)
from backend.prospective_dispatch import (
    _decision_input,
    _decision_loss_gradient as _prospective_decision_gradient,
    build_prospective_scenarios,
    finite_difference_decision_gradient as prospective_finite_difference,
)
from backend.domain import generate_routes
from backend.learning_dispatch import _route_id


class QuantumLearningTests(unittest.TestCase):
    def test_statevector_returns_bounded_expectation_and_nonconstant_features(self):
        a = expectation_z0((0.1, 0.2), (0.31, -0.47))
        b = expectation_z0((0.8, 0.7), (0.31, -0.47))
        self.assertGreaterEqual(a, -1.0)
        self.assertLessEqual(a, 1.0)
        self.assertNotAlmostEqual(a, b, places=6)

    def test_decision_gradient_matches_centered_finite_difference(self):
        rows, _ = _make_examples()
        sample = rows[:8]
        theta = (0.31, -0.47)
        analytic = decision_loss_gradient(theta, sample)
        numeric = finite_difference_gradient(theta, sample, 1e-5)
        self.assertTrue(all(abs(v) > 1e-4 for v in analytic))
        for x, y in zip(analytic, numeric):
            self.assertAlmostEqual(x, y, delta=2e-7)

    def test_probability_bounds_and_study_is_deterministic(self):
        p = risk_probability((0.3, 0.8), (0.31, -0.47))
        self.assertTrue(0.0 < p < 1.0)
        first, second = run_study(steps=40), run_study(steps=40)
        self.assertEqual(first, second)
        self.assertEqual(first["data"]["split_counts"], {"train": 40, "validation": 12, "test": 12})
        self.assertEqual(set(first["test_metrics"]), {"decision_focused_vqc", "predictive_only_vqc", "classical_logistic"})

    def test_boundary_rejects_bad_dimension_and_unbounded_steps(self):
        with self.assertRaises(ValueError):
            expectation_z0((0.2,), (0.1, 0.2))
        with self.assertRaises(ValueError):
            run_study(steps=2001)

    def test_dispatch_decision_gradient_matches_finite_differences(self):
        base = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
        train = [row for row in build_scenarios(base) if row.split == "train"][:2]
        theta = (0.31, -0.47)
        analytic = _decision_objective_and_gradient(theta, train)[1]
        numeric = finite_difference_decision_gradient(theta, train, 1e-5)
        self.assertTrue(all(abs(value) > 1e-5 for value in analytic))
        for left, right in zip(analytic, numeric):
            self.assertAlmostEqual(left, right, delta=2e-7)

    def test_dispatch_models_share_frozen_test_scenarios_and_validate_plans(self):
        base = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
        result = run_dispatch_study(base, steps=10)
        outcomes = result["test_outcomes"]
        self.assertEqual(set(outcomes), {"decision_focused_vqc", "predictive_only_vqc", "classical_logistic"})
        scenario_ids = [row["scenario_id"] for row in outcomes["decision_focused_vqc"]]
        self.assertEqual(len(scenario_ids), 4)
        for rows in outcomes.values():
            self.assertEqual([row["scenario_id"] for row in rows], scenario_ids)
            self.assertTrue(all(row["validation"]["valid"] and row["validation"]["feasible"] for row in rows))
            self.assertTrue(all(row["validation"]["recomputed"]["objective"] is not None for row in rows))
        self.assertEqual(result["split"]["train"][0]["scenario_id"], "traffic-00")
        self.assertEqual(result["split"]["test"][-1]["scenario_id"], "traffic-15")

    def test_prospective_protocol_groups_scenarios_and_hides_nontraining_labels(self):
        base = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
        scenarios = build_prospective_scenarios(base)
        scenarios_again = build_prospective_scenarios(base)
        counts = {name: sum(row.split == name for row in scenarios) for name in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 12, "validation": 6, "test": 6})
        self.assertEqual([(row.scenario_id, row.split, row.data_hash) for row in scenarios],
                         [(row.scenario_id, row.split, row.data_hash) for row in scenarios_again])
        test_scenario = next(row for row in scenarios if row.split == "test")
        visible = _decision_input(test_scenario)
        self.assertFalse(hasattr(visible, "realized"))
        self.assertTrue(all(row.train_label is None for row in visible.routes))
        validation = next(row for row in scenarios if row.split == "validation")
        self.assertTrue(all(row.train_label is None for row in validation.routes))
        self.assertNotIn("scenario_id", visible.forecast)
        self.assertNotIn("split", visible.forecast)

    def test_prospective_decision_gradient_matches_finite_difference_on_train_groups(self):
        base = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
        train = [row for row in build_prospective_scenarios(base) if row.split == "train"][:2]
        theta = (0.31, -0.47)
        analytic = _prospective_decision_gradient(theta, train)[1]
        numeric = prospective_finite_difference(theta, train, 1e-5)
        self.assertTrue(all(abs(value) > 1e-5 for value in analytic))
        for left, right in zip(analytic, numeric):
            self.assertAlmostEqual(left, right, delta=2e-7)

    def test_prospective_sla_label_uses_lateness_even_when_penalty_is_zero(self):
        """SLA event labels reflect lateness even when penalty is zero."""
        base = json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())
        for order in base["orders"]:
            order["penalty"] = 0
            order["window"] = [0, 0]
        for record in build_prospective_scenarios(base):
            if record.split != "train":
                continue
            actual = {_route_id(route): route for route in generate_routes(record.realized)}
            for row in record.routes:
                expected_event = int(any(stop["lateness"] > 0 for stop in actual[row.route_id]["stops"]))
                self.assertEqual(row.train_label, expected_event,
                                 f"{record.scenario_id}/{row.route_id}: late route mislabeled due to zero penalty")


if __name__ == "__main__":
    unittest.main()
