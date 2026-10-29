import json
from pathlib import Path
import unittest
from unittest.mock import patch

from backend.learning_inference import InferenceError, predict_dispatch
from backend.quantum_learning import risk_probability
from backend.qpiai_vqc import sdk_versions


def _scenario():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "dispatch_tiny.json").read_text(encoding="utf-8"))


def _four_order_scenario():
    scenario = _scenario()
    scenario["orders"].append({"id": "X", "demand": 1, "window": [0, 9], "service": 1, "penalty": 2})
    locations = ["D", "A", "B", "C", "X"]
    for origin in locations:
        scenario["travel"].setdefault(origin, {})
        for destination in locations:
            if destination not in scenario["travel"][origin]:
                scenario["travel"][origin][destination] = 0 if origin == destination else 2
    return scenario


class LearningInferenceTests(unittest.TestCase):
    def test_forecast_only_inference_returns_valid_plan_and_provenance(self):
        scenario = _scenario()
        result = predict_dispatch(scenario, scenario, [0.31, -0.47], model_version="synthetic-vector-test-v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_type"], "local_cpu_exact_statevector")
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(result["validation"]["feasible"])
        self.assertAlmostEqual(result["validation"]["recomputed"]["objective"], result["plan"]["objective"])
        self.assertEqual(result["model"]["version"], "synthetic-vector-test-v1")
        self.assertEqual(len(result["model"]["parameter_vector_sha256"]), 64)
        self.assertEqual(result["model"]["version_source"], "caller-supplied identifier; not registry-verified")
        self.assertEqual(result["model"]["parameter_source"], "caller-supplied two-angle vector; training origin unverified")
        self.assertEqual(result["model"]["training_provenance"], "unverified caller-supplied vector")
        self.assertEqual(result["model"]["calibration"], "unverified; inference has no calibration metadata")
        self.assertEqual(result["surrogate"]["currency_calibration"], "none; tiny-domain proxy units only")
        self.assertAlmostEqual(
            result["estimated_plan_cost"],
            sum(row["forecast_travel_cost"] + row["estimated_exposure"] for row in result["route_predictions"]),
        )
        for row in result["route_predictions"]:
            self.assertAlmostEqual(row["risk_probability"], risk_probability(row["features"], [0.31, -0.47]))
        self.assertEqual(sorted(oid for route in result["plan"]["routes"] for oid in route["order_ids"]), ["A", "B", "C"])
        self.assertNotIn("realized", result)
        self.assertEqual(result, predict_dispatch(scenario, scenario, [0.31, -0.47], model_version="synthetic-vector-test-v1"))

    def test_inference_accepts_distinct_forecast_travel_but_same_problem(self):
        nominal = _scenario()
        forecast = _scenario()
        for origin, edges in forecast["travel"].items():
            for destination, value in edges.items():
                if origin != destination:
                    edges[destination] = value * 1.2
        result = predict_dispatch(forecast, nominal, [-0.0925538742, -0.4402943343], model_version="prospective-synthetic-v1")
        self.assertTrue(result["validation"]["valid"])
        self.assertTrue(all(0 <= row["risk_probability"] <= 1 for row in result["route_predictions"]))

    def test_invalid_model_parameters_have_stable_codes(self):
        scenario = _scenario()
        cases = [([0.1], "theta"), ([0.1, float("nan")], "theta[1]"), ([0.1, 10.01], "theta[1]")]
        for theta, expected_field in cases:
            with self.subTest(theta=theta):
                with self.assertRaises(InferenceError) as caught:
                    predict_dispatch(scenario, scenario, theta, model_version="m1")
                self.assertEqual(caught.exception.code, "invalid_model_parameters")
                self.assertEqual(caught.exception.details[0]["field"], expected_field)

    def test_problem_mismatch_and_unknown_fields_are_rejected(self):
        nominal, forecast = _scenario(), _scenario()
        forecast["orders"][0]["window"] = [0, 3]
        with self.assertRaises(InferenceError) as mismatch:
            predict_dispatch(forecast, nominal, [0.1, 0.2], model_version="m1")
        self.assertEqual(mismatch.exception.code, "invalid_scenario")
        unknown = _scenario()
        unknown["realized"] = {"travel": "must-not-be-accepted"}
        with self.assertRaises(InferenceError) as extra:
            predict_dispatch(unknown, nominal, [0.1, 0.2], model_version="m1")
        self.assertEqual(extra.exception.code, "invalid_scenario")

    def test_infeasible_and_unsupported_scenarios_return_stable_errors(self):
        infeasible = _scenario()
        for order in infeasible["orders"]:
            order["demand"] = 100
        with self.assertRaises(InferenceError) as caught:
            predict_dispatch(infeasible, infeasible, [0.1, 0.2], model_version="m1")
        self.assertEqual(caught.exception.code, "infeasible")
        large = _four_order_scenario()
        with self.assertRaises(InferenceError) as unsupported:
            predict_dispatch(large, large, [0.1, 0.2], model_version="m1")
        self.assertEqual(unsupported.exception.code, "unsupported_capability")

    def test_model_version_is_required(self):
        scenario = _scenario()
        with self.assertRaises(InferenceError) as caught:
            predict_dispatch(scenario, scenario, [0.1, 0.2], model_version="")
        self.assertEqual(caught.exception.code, "invalid_model_parameters")

    def test_qpiai_local_matches_reference_or_reports_sdk_unavailable(self):
        scenario = _scenario()
        if sdk_versions().get("qpiai-quantum") == "unavailable":
            with self.assertRaises(InferenceError) as caught:
                predict_dispatch(scenario, scenario, [0.31, -0.47], model_version="m1",
                                 execution_backend="qpiai_local")
            self.assertEqual(caught.exception.code, "unsupported_capability")
            return
        reference = predict_dispatch(scenario, scenario, [0.31, -0.47], model_version="m1")
        actual = predict_dispatch(scenario, scenario, [0.31, -0.47], model_version="m1",
                                  execution_backend="qpiai_local")
        self.assertEqual(actual["execution_type"], "local_qpiai_statevector")
        self.assertEqual(actual["model"]["execution_backend"], "qpiai_local")
        self.assertNotEqual(actual["model"]["execution_backend_versions"].get("qpiai-quantum"), "unavailable")
        self.assertEqual(
            [(r["vehicle_id"], r["order_ids"]) for r in actual["plan"]["routes"]],
            [(r["vehicle_id"], r["order_ids"]) for r in reference["plan"]["routes"]],
        )
        self.assertAlmostEqual(actual["estimated_plan_cost"], reference["estimated_plan_cost"], delta=1e-10)
        for sdk_row, ref_row in zip(actual["route_predictions"], reference["route_predictions"]):
            self.assertEqual(sdk_row["route_id"], ref_row["route_id"])
            self.assertAlmostEqual(sdk_row["risk_probability"], ref_row["risk_probability"], delta=1e-10)
        self.assertTrue(actual["validation"]["valid"])

    def test_unknown_backend_and_sdk_failure_never_fall_back(self):
        scenario = _scenario()
        with self.assertRaises(InferenceError) as unsupported:
            predict_dispatch(scenario, scenario, [0.1, 0.2], model_version="m1", execution_backend="silent")
        self.assertEqual(unsupported.exception.code, "invalid_execution_backend")
        if sdk_versions().get("qpiai-quantum") != "unavailable":
            with patch("backend.qpiai_vqc.sdk_risk_probability", side_effect=RuntimeError("private detail")):
                with self.assertRaises(InferenceError) as failed:
                    predict_dispatch(scenario, scenario, [0.1, 0.2], model_version="m1",
                                     execution_backend="qpiai_local")
            self.assertEqual(failed.exception.code, "execution_failure")
            self.assertNotIn("private detail", failed.exception.message)

    def test_financial_terms_use_currency_estimates_and_independent_forecast_validation(self):
        scenario = {
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
        terms = {"currency": "USD", "operating_cost_per_travel_minute": 1.0,
                 "penalty_rate_unit": "currency_per_late_minute"}
        result = predict_dispatch(scenario, scenario, [0, 0], model_version="unverified-v",
                                  financial_terms=terms)
        self.assertEqual(result["plan"]["routes"][0]["order_ids"], ["B", "A"])
        risk = result["route_predictions"][0]["risk_probability"]
        self.assertAlmostEqual(result["route_predictions"][0]["estimated_transport_cost"], 0.5)
        self.assertAlmostEqual(result["estimated_exposure"], risk * 2.0)
        self.assertAlmostEqual(result["estimated_plan_cost"], 0.5 + risk * 2.0)
        self.assertEqual(result["estimated_cost_currency"], "USD")
        self.assertAlmostEqual(result["forecast_financial_objective"]["travel_minutes"], 0.5)
        self.assertAlmostEqual(result["forecast_financial_objective"]["late_penalty_cost"], 3.0)
        self.assertAlmostEqual(result["forecast_financial_objective"]["total_cost"], 3.5)
        self.assertTrue(result["financial_validation"]["valid"])
        self.assertIn("not known to have been trained for the supplied financial rates",
                      result["model_suitability_warning"])
        self.assertIn("one-minute", result["financial_estimate"]["exposure_assumption"])

    def test_financial_rate_can_change_selected_route_and_missing_terms_preserve_legacy(self):
        scenario = {
            "depot": "D", "vehicles": [{"id": "V1", "capacity": 2}],
            "orders": [{"id": "A", "demand": 1, "window": [0, 1], "service": 0, "penalty": 2},
                       {"id": "B", "demand": 1, "window": [0, 100], "service": 2, "penalty": 0}],
            "travel": {"D": {"D": 0, "A": 1, "B": 0}, "A": {"D": 0, "A": 0, "B": 9},
                       "B": {"D": 0, "A": 0.5, "B": 0}},
        }
        base = predict_dispatch(scenario, scenario, [0, 0], model_version="unverified-v")
        rate_zero = predict_dispatch(scenario, scenario, [0, 0], model_version="unverified-v",
                                     financial_terms={"currency": "USD", "operating_cost_per_travel_minute": 0,
                                                      "penalty_rate_unit": "currency_per_late_minute"})
        rate_one = predict_dispatch(scenario, scenario, [0, 0], model_version="unverified-v",
                                    financial_terms={"currency": "USD", "operating_cost_per_travel_minute": 1,
                                                     "penalty_rate_unit": "currency_per_late_minute"})
        self.assertEqual([r["order_ids"] for r in rate_zero["plan"]["routes"]], [["A", "B"]])
        self.assertEqual([r["order_ids"] for r in rate_one["plan"]["routes"]], [["B", "A"]])
        self.assertNotIn("financial_terms", base)
        self.assertNotIn("financial_validation", base)
        self.assertAlmostEqual(base["estimated_plan_cost"], sum(
            row["forecast_travel_cost"] + row["estimated_exposure"] for row in base["route_predictions"]))

    def test_financial_terms_are_required_and_tampered_validation_fails(self):
        scenario = _scenario()
        terms = {"currency": "USD", "operating_cost_per_travel_minute": 1.0,
                 "penalty_rate_unit": "currency_per_late_minute"}
        malformed = [None, {}, {**terms, "penalty_rate_unit": "unknown"},
                     {**terms, "operating_cost_per_travel_minute": float("nan")},
                     {**terms, "currency": "usd"}, {**terms, "extra": True}]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(InferenceError) as caught:
                predict_dispatch(scenario, scenario, [0.1, 0.2], model_version="m1", financial_terms=value)
            self.assertEqual(caught.exception.code, "invalid_financial_terms")

        result = predict_dispatch(scenario, scenario, [0.1, 0.2], model_version="m1", financial_terms=terms)
        from backend.financial import validate_financial_plan
        tampered = dict(result["forecast_financial_objective"])
        tampered["total_cost"] += 1
        check = validate_financial_plan(scenario, result["plan"], tampered,
                                        currency="USD", operating_cost_per_travel_minute=1.0,
                                        penalty_rate_unit="currency_per_late_minute")
        self.assertFalse(check["valid"])
        self.assertIn("total_cost mismatch", check["errors"])

    def test_hard_window_flags_are_preserved_and_forecast_plans_are_feasible(self):
        scenario = _scenario()
        for order in scenario["orders"]:
            order["hard_window"] = True
        backends = ["reference"]
        if sdk_versions().get("qpiai-quantum") != "unavailable":
            backends.append("qpiai_local")
        for backend in backends:
            with self.subTest(backend=backend):
                result = predict_dispatch(scenario, scenario, [0.31, -0.47],
                                          model_version="hard-window-test", execution_backend=backend)
                self.assertTrue(result["validation"]["valid"])
                self.assertTrue(all(stop["lateness"] <= 0
                                    for route in result["plan"]["routes"] for stop in route["stops"]))

    def test_hard_window_boolean_and_nominal_forecast_flag_mismatch_are_rejected(self):
        forecast, nominal = _scenario(), _scenario()
        forecast["orders"][0]["hard_window"] = "true"
        with self.assertRaises(InferenceError) as malformed:
            predict_dispatch(forecast, nominal, [0.1, 0.2], model_version="m1")
        self.assertEqual(malformed.exception.code, "invalid_scenario")

        forecast, nominal = _scenario(), _scenario()
        forecast["orders"][0]["hard_window"] = True
        with self.assertRaises(InferenceError) as mismatch:
            predict_dispatch(forecast, nominal, [0.1, 0.2], model_version="m1")
        self.assertEqual(mismatch.exception.code, "invalid_scenario")
        self.assertEqual(mismatch.exception.details[0]["field"], "orders[0].hard_window")

    def test_hard_window_infeasibility_is_explicit_and_nominal_is_feature_only(self):
        forecast, nominal = _scenario(), _scenario()
        forecast["orders"][0]["hard_window"] = True
        forecast["orders"][0]["window"] = [0, 0]
        nominal["orders"][0]["hard_window"] = True
        nominal["orders"][0]["window"] = [0, 0]
        with self.assertRaises(InferenceError) as infeasible:
            predict_dispatch(forecast, nominal, [0.1, 0.2], model_version="m1")
        self.assertEqual(infeasible.exception.code, "infeasible")

        # Nominal travel is used for feature encoding only. The forecast plan
        # remains hard-feasible and is independently checked against forecast.
        forecast, nominal = _scenario(), _scenario()
        forecast["orders"][0]["hard_window"] = True
        nominal["orders"][0]["hard_window"] = True
        nominal["travel"]["D"]["A"] = 100
        result = predict_dispatch(forecast, nominal, [0.1, 0.2], model_version="m1")
        self.assertTrue(result["validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
