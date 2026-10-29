from __future__ import annotations

import copy
import unittest

from backend.financial import (FinancialError, solve_financial_dispatch,
                               validate_financial_plan)


def tradeoff_scenario():
    return {
        "depot": "D",
        "vehicles": [{"id": "V1", "capacity": 2}],
        "orders": [
            {"id": "A", "demand": 1, "window": [0, 1], "service": 0, "penalty": 2},
            {"id": "B", "demand": 1, "window": [0, 100], "service": 2, "penalty": 0},
        ],
        "travel": {
            "D": {"D": 0, "A": 1, "B": 0},
            "A": {"D": 0, "A": 0, "B": 9},
            "B": {"D": 0, "A": 0.5, "B": 0},
        },
    }


def solve(scenario, rate):
    return solve_financial_dispatch(scenario, currency="USD",
                                    operating_cost_per_travel_minute=rate,
                                    penalty_rate_unit="currency_per_late_minute")


class FinancialTests(unittest.TestCase):
    def test_hand_computable_route_selection_changes_with_transport_rate(self):
        scenario = tradeoff_scenario()
        low_transport = solve(scenario, 0.1)
        high_transport = solve(scenario, 1.0)
        self.assertEqual(low_transport["plan"]["routes"][0]["order_ids"], ["A", "B"])
        self.assertEqual(high_transport["plan"]["routes"][0]["order_ids"], ["B", "A"])
        self.assertAlmostEqual(low_transport["financial_objective"]["travel_minutes"], 10.0)
        self.assertAlmostEqual(low_transport["financial_objective"]["late_penalty_cost"], 0.0)
        self.assertAlmostEqual(low_transport["financial_objective"]["total_cost"], 1.0)
        self.assertAlmostEqual(high_transport["financial_objective"]["travel_minutes"], 0.5)
        self.assertAlmostEqual(high_transport["financial_objective"]["late_penalty_cost"], 3.0)
        self.assertAlmostEqual(high_transport["financial_objective"]["total_cost"], 3.5)
        self.assertTrue(high_transport["proxy_validation"]["valid"])
        self.assertTrue(high_transport["financial_validation"]["valid"])
        self.assertEqual(high_transport["financial_objective"]["currency"], "USD")

    def test_currency_rate_and_penalty_unit_are_explicit_and_bounded(self):
        scenario = tradeoff_scenario()
        terms = {"currency": "USD", "operating_cost_per_travel_minute": 1.0,
                 "penalty_rate_unit": "currency_per_late_minute"}
        for key, value, expected in [
            ("currency", "usd", "invalid_currency"),
            ("currency", "US", "invalid_currency"),
            ("operating_cost_per_travel_minute", -1.0, "invalid_operating_rate"),
            ("operating_cost_per_travel_minute", float("nan"), "invalid_operating_rate"),
            ("penalty_rate_unit", "unspecified", "invalid_penalty_rate_unit"),
        ]:
            supplied = dict(terms, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(FinancialError) as caught:
                solve_financial_dispatch(scenario, **supplied)
            self.assertEqual(caught.exception.code, expected)

    def test_tampered_reported_money_fails_raw_input_recomputation(self):
        scenario = tradeoff_scenario()
        result = solve(scenario, 1.0)
        financial = copy.deepcopy(result["financial_objective"])
        financial["total_cost"] += 100
        checked = validate_financial_plan(scenario, result["plan"], financial,
                                          currency="USD", operating_cost_per_travel_minute=1.0,
                                          penalty_rate_unit="currency_per_late_minute")
        self.assertFalse(checked["valid"])
        self.assertIn("total_cost mismatch", checked["errors"])
        self.assertAlmostEqual(checked["recomputed_financial"]["total_cost"], 3.5)

    def test_infeasible_and_work_budget_exhaustion_fail_without_partial_plan(self):
        infeasible = tradeoff_scenario()
        for order in infeasible["orders"]:
            order["demand"] = 100
        with self.assertRaises(FinancialError) as caught:
            solve(infeasible, 1.0)
        self.assertEqual(caught.exception.code, "infeasible")

        orders = [{"id": oid, "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}
                  for oid in "ABCDE"]
        locations = ["D", *"ABCDE"]
        broad = {"depot": "D", "vehicles": [{"id": f"V{i}", "capacity": 1} for i in range(8)],
                 "orders": orders,
                 "travel": {origin: {destination: (0 if origin == destination else 1)
                                     for destination in locations} for origin in locations}}
        with self.assertRaises(FinancialError) as budget:
            solve(broad, 1.0)
        self.assertEqual(budget.exception.code, "work_budget_exceeded")
        self.assertNotIn("plan", budget.exception.as_dict())


if __name__ == "__main__":
    unittest.main()
