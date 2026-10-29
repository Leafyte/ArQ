"""Run the deterministic four-variable ordinary/warm-start QAOA comparison."""
from __future__ import annotations

import json
import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.qubo import build_route_qubo, enumerate_qubo, qubo_to_ising
from backend.qaoa import continuous_relaxation, run_qaoa


OUT = Path(__file__).with_name("tiny_results.json")
CANDIDATES = [
    {"id": "A_ab", "vehicle": "A", "stops": ["s1", "s2"], "cost": 4.0},
    {"id": "A_a", "vehicle": "A", "stops": ["s1"], "cost": 2.0},
    {"id": "B_b", "vehicle": "B", "stops": ["s2"], "cost": 2.0},
    {"id": "C_ab", "vehicle": "C", "stops": ["s1", "s2"], "cost": 5.0},
]


def summarize(result: dict, exact_cost: float) -> dict:
    feasible = [s for s in result["raw_samples"] if s["validation"]["feasible"]]
    repaired = [s["repair"] for s in result["raw_samples"] if s["repair"] and s["repair"]["validation"]["feasible"]]
    feasible_costs = [s["validation"]["business_cost"] for s in feasible]
    repaired_costs = [s["validation"]["business_cost"] for s in repaired]
    return {
        "mode": result["mode"], "backend": result["backend"], "seed": 20261028,
        "shots": result["shots"], "expected_qubo_energy": result["expected_qubo_energy"],
        "angles": result["angles"], "raw_counts": result["raw_counts"],
        "raw_sample_feasible_count": len(feasible),
        "raw_sample_feasibility_rate": len(feasible) / result["shots"],
        "best_raw_feasible_business_cost": min(feasible_costs) if feasible_costs else None,
        "best_raw_feasible_regret": min(feasible_costs) - exact_cost if feasible_costs else None,
        "repair_feasible_count": len(repaired),
        "best_repaired_business_cost": min(repaired_costs) if repaired_costs else None,
        "relaxation": result["relaxation"],
        "samples": result["raw_samples"],
    }


def main() -> None:
    q = build_route_qubo(CANDIDATES, ["s1", "s2"], coverage_penalty=20, vehicle_penalty=15)
    rows = enumerate_qubo(q)
    feasible = [row for row in rows if row["feasible"]]
    exact_cost = min(row["business_cost"] for row in feasible)
    relaxation = continuous_relaxation(q)
    common = {"seed": 20261028, "shots": 512, "angle_grid": 10}
    standard = run_qaoa(q, warm_start=False, **common)
    warm = run_qaoa(q, warm_start=True, relaxation=relaxation, **common)
    offset, h, j = qubo_to_ising(q)
    report = {
        "experiment_id": "tiny-route-qaoa-20261028-v1",
        "execution_type": "local_simulator",
        "simulator": "pure-Python statevector reference implementation; no QpiAI package/backend used",
        "python": sys.version,
        "platform": platform.platform(),
        "fixture_type": "deterministic synthetic illustration, not operational data",
        "vendor_sdk_probe": {
            "qpiai_quantum_importable": importlib.util.find_spec("qpiai_quantum") is not None,
            "qpiai_opt_importable": importlib.util.find_spec("qpiai_opt") is not None,
            "qcloud_or_opt_api_calls": 0,
            "qpu_jobs": 0,
            "credential_access": "not attempted",
        },
        "variable_order": list(q.variable_ids), "candidate_routes": CANDIDATES,
        "required_stops": list(q.required_stops), "coverage_penalty": q.coverage_penalty,
        "at_most_one_vehicle_penalty": q.vehicle_penalty,
        "qubo": {"linear": list(q.linear), "quadratic": {f"{i},{k}": v for (i, k), v in q.quadratic.items()}, "offset": q.offset},
        "ising": {"convention": "E=c+sum_i h_i z_i+sum_ij J_ij z_i z_j; x=(1-z)/2", "offset": offset, "h": h,
                  "J": {f"{i},{k}": v for (i, k), v in j.items()}, "exhaustive_equivalence": True},
        "exhaustive": {"bitstrings_checked": len(rows), "feasible_bitstrings": len(feasible),
                       "optimal_business_cost": exact_cost,
                       "optimal_selections": [r["selected_route_ids"] for r in feasible if r["business_cost"] == exact_cost]},
        "comparison": {"standard": summarize(standard.__dict__, exact_cost),
                       "warm_start": summarize(warm.__dict__, exact_cost)},
        "limits": ["single-layer QAOA", "coarse equal 10x10 angle grid, no continuous optimizer",
                   "exact statevector, finite-shot sampling", "no operational validity beyond coverage and vehicle uniqueness",
                   "not QpiAI QCloud, QpiAI Opt, authenticated service, or QPU evidence"],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({mode: {"feasible": data["raw_sample_feasible_count"],
                                 "best_cost": data["best_raw_feasible_business_cost"]}
                      for mode, data in report["comparison"].items()}, indent=2))


if __name__ == "__main__":
    main()
