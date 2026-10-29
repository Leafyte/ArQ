from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from backend.domain import solve_exact
from backend.risk import RiskError, evaluate_plan_risk


def problem(penalty: float = 10.0):
    scenario = {
        "depot": "D",
        "vehicles": [{"id": "V", "capacity": 1}],
        "orders": [{"id": "A", "demand": 1, "window": [0, 2], "service": 0, "penalty": penalty}],
        "travel": {"D": {"D": 0, "A": 1}, "A": {"D": 1, "A": 0}},
    }
    low = {"D": {"D": 0, "A": 1}, "A": {"D": 1, "A": 0}}
    high = {"D": {"D": 0, "A": 3}, "A": {"D": 1, "A": 0}}
    return scenario, solve_exact(scenario), low, high


class RiskTests(unittest.TestCase):
    def test_hand_calculated_empirical_risk_and_fractional_cvar(self):
        scenario, plan, low, high = problem()
        cases = [{"scenario_id": "on-time", "travel": low},
                 {"scenario_id": "late", "travel": high}]
        result = evaluate_plan_risk(scenario, scenario, plan, cases, cvar_alpha=0.75)
        summary = result["summary"]
        self.assertEqual(result["estimator"]["scenario_count"], 2)
        self.assertEqual(summary["empirical_probability_any_late"], 0.5)
        self.assertEqual(summary["empirical_late_stop_frequency"], 0.5)
        self.assertEqual(summary["per_order_late_frequency"], {"A": 0.5})
        self.assertEqual(summary["expected_travel_cost"], 3.0)
        self.assertEqual(summary["expected_penalty_exposure"], 5.0)
        self.assertEqual(summary["expected_proxy_objective"], 8.0)
        self.assertEqual(result["cvar"]["system_wide_penalty_exposure"], 10.0)
        self.assertEqual(result["cvar"]["system_wide_proxy_objective"], 14.0)
        self.assertEqual(result["scenarios"][1]["routes"][0]["stops"][0]["lateness"], 1.0)
        self.assertTrue(all(row["validation"]["valid"] for row in result["scenarios"]))

    def test_zero_penalty_still_counts_late_events(self):
        scenario, plan, low, high = problem(penalty=0.0)
        result = evaluate_plan_risk(scenario, scenario, plan,
                                    [{"scenario_id": "normal", "travel": low},
                                     {"scenario_id": "late-zero-penalty", "travel": high}])
        self.assertEqual(result["summary"]["empirical_probability_any_late"], 0.5)
        self.assertEqual(result["summary"]["expected_penalty_exposure"], 0.0)

    def test_hard_window_violation_keeps_empirical_metrics_but_marks_replay_infeasible(self):
        scenario, plan, low, high = problem()
        scenario["orders"][0]["hard_window"] = True
        # The submitted plan is feasible under the nominal/forecast matrix.
        result = evaluate_plan_risk(
            scenario, scenario, plan,
            [{"scenario_id": "within-window", "travel": low},
             {"scenario_id": "hard-window-breach", "travel": high}],
        )
        breach = result["scenarios"][1]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["empirical_probability_any_late"], 0.5)
        self.assertEqual(result["summary"]["empirical_probability_hard_window_violation"], 0.5)
        self.assertFalse(breach["validation"]["valid"])
        self.assertFalse(breach["feasible"])
        self.assertTrue(breach["hard_window_violation"])
        self.assertTrue(any("hard_window" in error for error in breach["validation"]["errors"]))
        self.assertEqual(breach["routes"][0]["stops"][0]["lateness"], 1.0)

    def test_unrelated_validation_error_containing_hard_window_phrase_fails_closed(self):
        scenario, plan, low, _ = problem()
        from backend import risk as risk_module

        real_validate = risk_module.validate_plan
        calls = 0

        def contaminated_validation(scenario_value, plan_value):
            nonlocal calls
            calls += 1
            result = real_validate(scenario_value, plan_value)
            if calls == 2:
                result["valid"] = False
                result["feasible"] = False
                result["errors"] = [
                    "routes[0].stops[0]: unrelated failure mentions "
                    "routes[0].stops[0].hard_window: service starts after hard close"
                ]
            return result

        with patch("backend.risk.validate_plan", side_effect=contaminated_validation):
            with self.assertRaises(RiskError) as caught:
                evaluate_plan_risk(
                    scenario, scenario, plan,
                    [{"scenario_id": "tampered-validation", "travel": low}],
                )
        self.assertEqual(caught.exception.code, "validation_failure")

    def test_invalid_realized_matrix_and_invalid_plan_fail_closed(self):
        scenario, plan, low, high = problem()
        invalid = copy.deepcopy(low)
        del invalid["D"]["A"]
        with self.assertRaises(RiskError) as bad_matrix:
            evaluate_plan_risk(scenario, scenario, plan, [{"scenario_id": "bad", "travel": invalid}])
        self.assertEqual(bad_matrix.exception.code, "invalid_scenario")
        altered = copy.deepcopy(plan)
        altered["objective"] += 1
        with self.assertRaises(RiskError) as bad_plan:
            evaluate_plan_risk(scenario, scenario, altered, [{"scenario_id": "ok", "travel": high}])
        self.assertEqual(bad_plan.exception.code, "invalid_plan")

    def test_repeated_matrix_under_distinct_ids_contributes_repeated_weight(self):
        scenario, plan, low, high = problem()
        result = evaluate_plan_risk(scenario, scenario, plan,
                                    [{"scenario_id": "on-time-1", "travel": low},
                                     {"scenario_id": "on-time-2", "travel": low},
                                     {"scenario_id": "late", "travel": high}])
        summary = result["summary"]
        self.assertEqual(result["estimator"]["scenario_count"], 3)
        self.assertEqual(summary["empirical_probability_any_late"], 1 / 3)
        self.assertAlmostEqual(summary["expected_travel_cost"], 8 / 3)
        self.assertAlmostEqual(summary["expected_penalty_exposure"], 10 / 3)
        self.assertEqual(result["scenarios"][0]["travel_matrix_sha256"],
                         result["scenarios"][1]["travel_matrix_sha256"])
        self.assertEqual(len({row["scenario_id"] for row in result["scenarios"]}), 3)

    def test_hashes_and_output_are_deterministic_and_inputs_are_unchanged(self):
        scenario, plan, low, high = problem()
        cases = [{"scenario_id": "s1", "travel": low}, {"scenario_id": "s2", "travel": high}]
        before = copy.deepcopy((scenario, plan, cases))
        first = evaluate_plan_risk(scenario, scenario, plan, cases, cvar_alpha=0.5)
        second = evaluate_plan_risk(scenario, scenario, plan, cases, cvar_alpha=0.5)
        self.assertEqual(first, second)
        self.assertEqual((scenario, plan, cases), before)
        self.assertEqual(len(first["provenance"]["realized_scenario_set_sha256"]), 64)
        self.assertEqual(len(first["scenarios"][0]["travel_matrix_sha256"]), 64)

    def test_cvar_alpha_is_bounded(self):
        scenario, plan, low, _ = problem()
        for alpha in (-0.1, 1.0, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(RiskError) as caught:
                evaluate_plan_risk(scenario, scenario, plan, [{"scenario_id": "s", "travel": low}],
                                   cvar_alpha=alpha)
            self.assertEqual(caught.exception.code, "invalid_cvar_alpha")


if __name__ == "__main__":
    unittest.main()
