"""Run QpiAI local warm-start on the same route QUBO as the local reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.domain import generate_routes
from backend.qaoa import continuous_relaxation, exhaustive_optimum, run_qaoa
from backend.qpiai_adapter import run_qpiai_warm_start
from backend.qubo import build_route_qubo, validate_selection
from backend.validator import validate_plan


HERE = Path(__file__).resolve().parent


def main() -> None:
    scenario = json.loads((HERE / "dispatch_fixture.json").read_text(encoding="utf-8"))
    route_pool = generate_routes(scenario)
    candidates = [{"id": f"{r['vehicle_id']}:{i}:" + ",".join(r["order_ids"]),
                   "vehicle": r["vehicle_id"], "stops": r["order_ids"], "cost": r["objective"]}
                  for i, r in enumerate(route_pool)]
    qubo = build_route_qubo(candidates, [o["id"] for o in scenario["orders"]])
    relaxation = continuous_relaxation(qubo)
    # QpiAI Opt's documented WarmStart clamps to [epsilon,1-epsilon]. Apply the
    # same vector to the local reference to keep state initialization aligned.
    aligned = [min(0.75, max(0.25, p)) for p in relaxation]
    local = run_qaoa(qubo, seed=20261028, shots=128, warm_start=True,
                     relaxation=aligned, angle_grid=6)
    sdk_result = run_qpiai_warm_start(
        qubo, aligned, gamma=local.angles["gamma"][0], beta=local.angles["beta"][0],
        shots=128, seed=20261028,
    )
    # Verify the SDK's printed bitstring direction from distinct product-state
    # probabilities with cost/mixer angles zero (so only warm-state preparation acts).
    probe_probabilities = [0.25 + i * 0.07 for i in range(qubo.n)]
    mapping_probe = run_qpiai_warm_start(
        qubo, probe_probabilities, gamma=0.0, beta=0.0, shots=2048, seed=20261029,
    )
    mapping_evidence = {"status": "unavailable"}
    if mapping_probe.get("status") == "success":
        counts = mapping_probe["raw_counts"]
        observed = [sum(count for bitstring, count in counts.items() if str(bitstring)[pos] == "1") / 2048
                    for pos in range(qubo.n)]
        direct_error = sum(abs(a-b) for a, b in zip(observed, probe_probabilities)) / qubo.n
        reverse_error = sum(abs(a-b) for a, b in zip(observed, reversed(probe_probabilities))) / qubo.n
        if abs(direct_error - reverse_error) >= 0.03:
            position_to_variable = (list(range(qubo.n)) if direct_error < reverse_error
                                    else list(reversed(range(qubo.n))))
            mapping_evidence = {"status": "verified_empirically", "position_to_variable_index": position_to_variable,
                                "expected_probabilities_by_variable": probe_probabilities,
                                "observed_one_probabilities_by_output_position": observed,
                                "mean_absolute_error_direct": direct_error,
                                "mean_absolute_error_reversed": reverse_error,
                                "shots": 2048, "angles": {"gamma": 0.0, "beta": 0.0}}
            decoded = []
            for raw_bits, count in sorted(sdk_result.get("raw_counts", {}).items()):
                raw_bits = str(raw_bits)
                if len(raw_bits) != qubo.n or any(c not in "01" for c in raw_bits):
                    continue
                bits = ["0"] * qubo.n
                for output_pos, variable_index in enumerate(position_to_variable):
                    bits[variable_index] = raw_bits[output_pos]
                variable_bits = "".join(bits)
                binary = [int(c) for c in variable_bits]
                route_plan = {"status": "feasible",
                              "routes": [route for route, bit in zip(route_pool, binary) if bit],
                              "objective": sum(route["objective"] for route, bit in zip(route_pool, binary) if bit),
                              "travel_cost": sum(route["travel_cost"] for route, bit in zip(route_pool, binary) if bit),
                              "lateness_exposure": sum(route["lateness_exposure"] for route, bit in zip(route_pool, binary) if bit)}
                decoded.append({"raw_sdk_bitstring": raw_bits, "variable_order_bitstring": variable_bits,
                                "count": count, "qubo_validation": validate_selection(qubo, binary),
                                "independent_domain_validation": validate_plan(scenario, route_plan),
                                "plan": route_plan})
            sdk_result["decoded_samples"] = decoded
            feasible_samples = [row for row in decoded if row["qubo_validation"]["feasible"]
                                and row["independent_domain_validation"]["valid"]]
            if feasible_samples:
                chosen = min(feasible_samples, key=lambda row: (row["qubo_validation"]["business_cost"],
                                                                  row["variable_order_bitstring"]))
                sdk_result["selected_sample"] = {
                    "source": "QpiAI local simulator raw measurement",
                    "raw_sdk_bitstring": chosen["raw_sdk_bitstring"],
                    "variable_order_bitstring": chosen["variable_order_bitstring"],
                    "observed_count": chosen["count"],
                    "selected_route_ids": chosen["qubo_validation"]["selected_route_ids"],
                    "business_cost": chosen["qubo_validation"]["business_cost"],
                    "plan": chosen["plan"],
                    "independent_validation": chosen["independent_domain_validation"],
                    "repair": None,
                }
            else:
                sdk_result["selected_sample"] = None
            sdk_result["bitstring_mapping"] = mapping_evidence
    if mapping_evidence["status"] != "verified_empirically" and sdk_result.get("status") == "success":
        sdk_result["bitstring_mapping"] = mapping_evidence
    exact = exhaustive_optimum(qubo)
    report = {
        "experiment_id": "qpiai-opt-route-qubo-warm-start-20261028",
        "fixture": "experiments_qaoa/dispatch_fixture.json",
        "fixture_type": "deterministic synthetic illustration",
        "same_route_qubo_as_dispatch_adapter": True,
        "variable_order": list(qubo.variable_ids),
        "candidate_routes": candidates,
        "qubo": {"linear": list(qubo.linear),
                 "quadratic": {f"{i},{j}": v for (i,j),v in qubo.quadratic.items()},
                 "offset": qubo.offset, "coverage_penalty": qubo.coverage_penalty,
                 "vehicle_penalty": qubo.vehicle_penalty},
        "continuous_relaxation": {"method": "projected_gradient_continuous_box_relaxation",
                                  "raw_vector": relaxation, "sdk_aligned_vector": aligned,
                                  "sdk_epsilon": 0.25},
        "exact_reference": exact,
        "local_reference": {"execution_type": local.backend, "shots": local.shots,
                            "angles": local.angles, "expected_qubo_energy": local.expected_qubo_energy,
                            "raw_counts": local.raw_counts,
                            "raw_feasible_count": local.feasible_samples},
        "qpiai": sdk_result,
        "qpiai_bitstring_mapping_probe": {k: v for k, v in mapping_probe.items() if k != "raw_counts"},
        "limits": ["QpiAI Opt local solver Client not invoked", "no credentials inspected",
                   "no API, cloud simulator, or QPU jobs submitted",
                   "QpiAI bitstring direction was empirically checked with distinct warm-state marginals before decoding",
                   "synthetic result is not a fleet deployment or quantum advantage claim"],
    }
    out = HERE / "qpiai_warm_start_results.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"status": sdk_result.get("status"),
                      "execution_type": sdk_result.get("execution_type"),
                      "backend": sdk_result.get("backend"),
                      "raw_counts": sdk_result.get("raw_counts"),
                      "error": sdk_result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
