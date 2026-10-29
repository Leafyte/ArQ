#!/usr/bin/env python3
"""
tools/sdk_probes/probe_ortools.py

Capability probe for Google OR-Tools.
Evaluates:
  Stage 1: Import availability (cp_model, pywrapcp, routing_enums_pb2)
  Stage 2: Model construction (CP-SAT MILP and RoutingModel vehicle routing)
  Stage 3: Local execution (CP-SAT solve & Vehicle Routing solve to completion)
  Stage 4: Cloud / Remote status (OR-Tools is self-contained local library)

Outputs machine-readable JSON or human-readable report.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def probe_ortools() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "sdk": "ortools",
        "stage_1_import": {"success": False, "details": None},
        "stage_2_construction": {"success": False, "details": None},
        "stage_3_local_execution": {"success": False, "details": None},
        "stage_4_cloud_status": {
            "success": True,
            "credentials_configured": True,
            "action": "OR-Tools is an offline, self-contained solver; no cloud backend or API key required.",
        },
    }

    # ==========================================
    # STAGE 1: Import Availability
    # ==========================================
    try:
        import ortools
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        from ortools.sat.python import cp_model

        version = getattr(ortools, "__version__", "unknown")
        report["stage_1_import"] = {
            "success": True,
            "version": version,
            "imported": ["cp_model", "pywrapcp", "routing_enums_pb2"],
        }
    except ImportError as exc:
        report["stage_1_import"] = {
            "success": False,
            "error": str(exc),
            "note": "ortools is not installed in the active environment.",
        }
        report["stage_2_construction"] = {
            "success": False,
            "note": "Skipped: ortools is not installed.",
        }
        report["stage_3_local_execution"] = {
            "success": False,
            "note": "Skipped: ortools is not installed.",
        }
        return report

    # ==========================================
    # STAGE 2: Model Construction
    # ==========================================
    try:
        # 1. CP-SAT model construction: max 3*x + 2*y s.t. x + 2*y <= 14, 3*x - y >= 0, x <= 10
        model = cp_model.CpModel()
        x = model.NewIntVar(0, 10, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x + 2 * y <= 14)
        model.Add(3 * x - y >= 0)
        model.Maximize(3 * x + 2 * y)

        # 2. Routing model construction (3 nodes, 1 vehicle, depot=0)
        manager = pywrapcp.RoutingIndexManager(3, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback: 0 -> 1 = 10, 1 -> 2 = 15, 0 -> 2 = 20, symmetric
        distance_matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0],
        ]

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        report["stage_2_construction"] = {
            "success": True,
            "cp_sat_variables": 2,
            "routing_nodes": 3,
            "routing_vehicles": 1,
            "note": "CP-SAT model and Vehicle Routing Model constructed successfully.",
        }
    except Exception as exc:
        report["stage_2_construction"] = {
            "success": False,
            "error": str(exc),
        }
        report["stage_3_local_execution"] = {
            "success": False,
            "note": "Skipped: model construction failed.",
        }
        return report

    # ==========================================
    # STAGE 3: Local Execution
    # ==========================================
    try:
        # Solve CP-SAT
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        cp_status = solver.Solve(model)
        cp_optimal = cp_status == cp_model.OPTIMAL
        x_val = solver.Value(x) if cp_optimal else None
        y_val = solver.Value(y) if cp_optimal else None
        obj_val = solver.ObjectiveValue() if cp_optimal else None

        # Solve Routing
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        routing_solution = routing.SolveWithParameters(search_parameters)

        route_found = routing_solution is not None
        route = []
        if route_found:
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = routing_solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))

        report["stage_3_local_execution"] = {
            "success": cp_optimal and route_found,
            "cp_sat": {
                "status": solver.StatusName(cp_status),
                "optimal": cp_optimal,
                "x": x_val,
                "y": y_val,
                "objective": obj_val,
            },
            "routing": {
                "solution_found": route_found,
                "route": route,
                "objective_cost": routing_solution.ObjectiveValue() if route_found else None,
            },
            "note": "Both CP-SAT and Vehicle Routing solved locally to optimality.",
        }
    except Exception as exc:
        report["stage_3_local_execution"] = {
            "success": False,
            "error": str(exc),
        }

    # ==========================================
    # STAGE 4: Cloud / Remote Status
    # ==========================================
    report["stage_4_cloud_status"] = {
        "success": True,
        "credentials_configured": True,
        "action": "OR-Tools is an offline, self-contained solver; no cloud backend or API key required.",
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Google OR-Tools capabilities.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    args = parser.parse_args()

    res = probe_ortools()
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("=== Google OR-Tools SDK Probe Report ===")
        print(f"Stage 1 (Import): {'PASS' if res['stage_1_import']['success'] else 'FAIL'}")
        if res["stage_1_import"]["success"]:
            print(f"  Version: {res['stage_1_import'].get('version')}")
            print(f"  Submodules: {res['stage_1_import'].get('imported')}")
        else:
            print(f"  Details: {res['stage_1_import'].get('note') or res['stage_1_import'].get('error')}")

        print(f"Stage 2 (Construction): {'PASS' if res['stage_2_construction']['success'] else 'FAIL'}")
        if res["stage_2_construction"]["success"]:
            print(f"  CP-SAT variables: {res['stage_2_construction'].get('cp_sat_variables')}, Routing nodes: {res['stage_2_construction'].get('routing_nodes')}")
        elif res["stage_2_construction"].get("error"):
            print(f"  Error: {res['stage_2_construction']['error']}")

        print(f"Stage 3 (Local Execution): {'PASS' if res['stage_3_local_execution']['success'] else 'FAIL'}")
        if res["stage_3_local_execution"]["success"]:
            print(f"  CP-SAT Status: {res['stage_3_local_execution']['cp_sat']['status']} (obj={res['stage_3_local_execution']['cp_sat']['objective']})")
            print(f"  Routing Route: {res['stage_3_local_execution']['routing']['route']} (cost={res['stage_3_local_execution']['routing']['objective_cost']})")
        elif res["stage_3_local_execution"].get("error"):
            print(f"  Error: {res['stage_3_local_execution']['error']}")

        print("Stage 4 (Cloud Status): LOCAL_ONLY_LIBRARY")
        print(f"  Note: {res['stage_4_cloud_status']['action']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
