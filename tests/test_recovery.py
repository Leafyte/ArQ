from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from backend.domain import generate_routes, solve_exact
from backend.recovery import RecoveryError, recover_dispatch
from backend.validator import validate_plan


def fixture() -> dict:
    return json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text())


def perturb_travel(scenario: dict, factor: float) -> dict:
    updated = copy.deepcopy(scenario)
    for origin, row in updated["travel"].items():
        for destination in row:
            if origin != destination:
                row[destination] *= factor
    return updated


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.original_scenario = fixture()
        self.original_plan = solve_exact(self.original_scenario)

    def test_valid_recovery_preserves_prefix_and_independently_validates(self):
        updated = perturb_travel(self.original_scenario, 1.2)
        route = self.original_plan["routes"][0]
        prefix = route["order_ids"][:1]
        result = recover_dispatch(self.original_scenario, self.original_plan, updated,
                                  {route["vehicle_id"]: prefix}, change_penalty=0.5)
        recovered_by_vehicle = {r["vehicle_id"]: r for r in result["recovered_plan"]["routes"]}
        self.assertEqual(recovered_by_vehicle[route["vehicle_id"]]["order_ids"][:1], prefix)
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(result["validation"], validate_plan(updated, result["recovered_plan"]))
        baseline = result["static_original_baseline"]
        self.assertTrue(baseline["available"])
        self.assertTrue(baseline["validation"]["valid"])
        self.assertEqual(baseline["validation"], validate_plan(updated, baseline["plan"]))
        self.assertIn("recomputed under updated inputs", baseline["comparison_basis"])
        self.assertIn("depot_replay", result["state_basis"])

    def test_original_plan_and_inputs_are_not_mutated(self):
        original_plan_before = copy.deepcopy(self.original_plan)
        scenario_before = copy.deepcopy(self.original_scenario)
        updated = perturb_travel(self.original_scenario, 1.1)
        result = recover_dispatch(self.original_scenario, self.original_plan, updated, {})
        result["original_plan"]["routes"].clear()
        self.assertEqual(self.original_plan, original_plan_before)
        self.assertEqual(self.original_scenario, scenario_before)
        self.assertEqual(updated, perturb_travel(self.original_scenario, 1.1))

    def test_locked_vehicle_unavailable_is_rejected(self):
        route = self.original_plan["routes"][0]
        updated = copy.deepcopy(self.original_scenario)
        updated["vehicles"] = [v for v in updated["vehicles"] if v["id"] != route["vehicle_id"]]
        with self.assertRaises(RecoveryError) as caught:
            recover_dispatch(self.original_scenario, self.original_plan, updated,
                             {route["vehicle_id"]: route["order_ids"][:1]})
        self.assertEqual(caught.exception.code, "locked_work_vehicle_unavailable")

    def test_infeasible_after_vehicle_unavailable_is_rejected(self):
        updated = copy.deepcopy(self.original_scenario)
        updated["vehicles"] = [v for v in updated["vehicles"] if v["id"] == "V1"]
        with self.assertRaises(RecoveryError) as caught:
            recover_dispatch(self.original_scenario, self.original_plan, updated, {})
        self.assertEqual(caught.exception.code, "recovery_infeasible")

    def test_invalid_disruption_and_unproven_prefix_are_rejected(self):
        invalid = copy.deepcopy(self.original_scenario)
        del invalid["travel"]["D"]["A"]
        with self.assertRaises(RecoveryError) as bad_scenario:
            recover_dispatch(self.original_scenario, self.original_plan, invalid, {})
        self.assertEqual(bad_scenario.exception.code, "invalid_updated_scenario")
        vehicle_id = self.original_plan["routes"][0]["vehicle_id"]
        with self.assertRaises(RecoveryError) as bad_prefix:
            recover_dispatch(self.original_scenario, self.original_plan, self.original_scenario,
                             {vehicle_id: ["not-the-original-first-stop"]})
        self.assertEqual(bad_prefix.exception.code, "locked_prefix_unproven")

    def test_search_stops_at_recovery_node_budget_without_partial_plan(self):
        orders = [
            {"id": oid, "demand": 1, "window": [0, 100], "service": 0, "penalty": 0}
            for oid in "ABCD"
        ]
        locations = ["D", *"ABCD"]
        scenario = {
            "depot": "D",
            "vehicles": [{"id": f"V{i:02d}", "capacity": 4} for i in range(16)],
            "orders": orders,
            "travel": {origin: {destination: (0 if origin == destination else 1)
                                for destination in locations} for origin in locations},
        }
        generated = generate_routes(scenario)
        chosen = [next(route for route in generated
                       if route["vehicle_id"] == f"V{i:02d}" and route["order_ids"] == [oid])
                  for i, oid in enumerate("ABCD")]
        plan = {"status": "feasible", "routes": chosen,
                "travel_cost": sum(route["travel_cost"] for route in chosen),
                "lateness_exposure": sum(route["lateness_exposure"] for route in chosen),
                "objective": sum(route["objective"] for route in chosen)}
        with self.assertRaises(RecoveryError) as caught:
            recover_dispatch(scenario, plan, scenario, {})
        self.assertEqual(caught.exception.code, "recovery_budget_exceeded")
        self.assertNotIn("recovered_plan", caught.exception.as_dict())


if __name__ == "__main__":
    unittest.main()
