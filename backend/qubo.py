"""Small, dependency-free route-selection QUBO with independent validation.

Variables are candidate routes in input order.  A feasible selection covers
each required stop exactly once and selects at most one route per vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Qubo:
    variable_ids: tuple[str, ...]
    linear: tuple[float, ...]
    quadratic: Mapping[tuple[int, int], float]
    offset: float
    required_stops: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    coverage_penalty: float
    vehicle_penalty: float

    @property
    def n(self) -> int:
        return len(self.variable_ids)

    def energy(self, bits: Sequence[int]) -> float:
        if len(bits) != self.n or any(not _binary_value(bit) for bit in bits):
            raise ValueError("bits must be a binary vector matching the QUBO size")
        return self.offset + sum(a * int(x) for a, x in zip(self.linear, bits)) + sum(
            value * int(bits[i]) * int(bits[j]) for (i, j), value in self.quadratic.items()
        )


def build_route_qubo(
    candidates: Sequence[Mapping[str, Any]],
    required_stops: Sequence[str],
    *,
    coverage_penalty: float | None = None,
    vehicle_penalty: float | None = None,
) -> Qubo:
    """Build cost + exact coverage + at-most-one-per-vehicle penalties.

    Candidate schema: ``{id, vehicle, stops, cost}``.  Penalties default to
    1 + the sum of absolute route costs, which exceeds any single objective
    improvement on these nonnegative-cost instances.
    """
    if not candidates:
        raise ValueError("at least one route candidate is required")
    stops = tuple(required_stops)
    if not stops or len(set(stops)) != len(stops):
        raise ValueError("required_stops must be non-empty and unique")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for c in candidates:
        if not all(k in c for k in ("id", "vehicle", "stops", "cost")):
            raise ValueError("each candidate needs id, vehicle, stops, and cost")
        cid, vehicle = str(c["id"]), str(c["vehicle"])
        route_stops = tuple(c["stops"])
        cost = float(c["cost"])
        if not cid or cid in ids or not vehicle or not isfinite(cost) or cost < 0:
            raise ValueError("candidate ids must be unique; vehicle and nonnegative finite cost are required")
        if not route_stops or len(set(route_stops)) != len(route_stops) or not set(route_stops) <= set(stops):
            raise ValueError(f"candidate {cid} has invalid or unknown stops")
        ids.add(cid)
        normalized.append({"id": cid, "vehicle": vehicle, "stops": route_stops, "cost": cost})
    default_penalty = 1.0 + sum(c["cost"] for c in normalized)
    pc = default_penalty if coverage_penalty is None else float(coverage_penalty)
    pv = default_penalty if vehicle_penalty is None else float(vehicle_penalty)
    if not isfinite(pc) or not isfinite(pv) or pc <= 0 or pv <= 0:
        raise ValueError("penalties must be positive finite values")

    linear = [c["cost"] for c in normalized]
    quad: dict[tuple[int, int], float] = {}
    offset = 0.0
    for stop in stops:
        indices = [i for i, c in enumerate(normalized) if stop in c["stops"]]
        if not indices:
            raise ValueError(f"no candidate covers required stop {stop!r}")
        # P (sum_i x_i - 1)^2 = P + sum_i -P*x_i + 2P*x_i*x_j
        offset += pc
        for i in indices:
            linear[i] -= pc
        for i, j in combinations(indices, 2):
            quad[i, j] = quad.get((i, j), 0.0) + 2 * pc
    by_vehicle: dict[str, list[int]] = {}
    for i, c in enumerate(normalized):
        by_vehicle.setdefault(c["vehicle"], []).append(i)
    for indices in by_vehicle.values():
        for i, j in combinations(indices, 2):
            quad[i, j] = quad.get((i, j), 0.0) + pv
    return Qubo(tuple(c["id"] for c in normalized), tuple(linear), quad, offset,
                stops, tuple(normalized), pc, pv)


def validate_selection(qubo: Qubo, bits: Sequence[int]) -> dict[str, Any]:
    """Recompute route costs, coverage, duplicates, and business feasibility."""
    if len(bits) != qubo.n or any(not _binary_value(b) for b in bits):
        raise ValueError("bits must be a binary vector matching the QUBO size")
    selected = [c for c, b in zip(qubo.candidates, bits) if int(b)]
    counts = {s: sum(s in c["stops"] for c in selected) for s in qubo.required_stops}
    vehicles = [c["vehicle"] for c in selected]
    coverage_ok = all(v == 1 for v in counts.values())
    vehicle_ok = len(set(vehicles)) == len(vehicles)
    return {
        "feasible": coverage_ok and vehicle_ok,
        "selected_route_ids": [c["id"] for c in selected],
        "coverage_counts": counts,
        "duplicate_vehicles": sorted(v for v in set(vehicles) if vehicles.count(v) > 1),
        "business_cost": sum(c["cost"] for c in selected),
        "coverage_ok": coverage_ok,
        "vehicle_ok": vehicle_ok,
    }


def _binary_value(value: Any) -> bool:
    return isinstance(value, (int, bool)) and value in (0, 1)


def decode_bitstring(qubo: Qubo, bitstring: str) -> dict[str, Any]:
    """Decode variable-order bit text and independently validate the selection."""
    if len(bitstring) != qubo.n or any(ch not in "01" for ch in bitstring):
        raise ValueError("bitstring must contain exactly one 0/1 character per route variable")
    bits = [int(ch) for ch in bitstring]
    return {"bitstring": bitstring, "variable_order": list(qubo.variable_ids),
            "qubo_energy": qubo.energy(bits), "validation": validate_selection(qubo, bits)}


def enumerate_qubo(qubo: Qubo) -> list[dict[str, Any]]:
    """Exhaustively cross-check energy against direct penalty recomputation."""
    rows = []
    for bits in product((0, 1), repeat=qubo.n):
        validation = validate_selection(qubo, bits)
        direct = validation["business_cost"]
        direct += qubo.coverage_penalty * sum((count - 1) ** 2 for count in validation["coverage_counts"].values())
        # at-most-one: each colliding pair incurs P, regardless of group size
        for vehicle in set(c["vehicle"] for c in qubo.candidates):
            n = sum(int(b) for c, b in zip(qubo.candidates, bits) if c["vehicle"] == vehicle)
            direct += qubo.vehicle_penalty * n * (n - 1) / 2
        energy = qubo.energy(bits)
        if abs(energy - direct) > 1e-8:
            raise AssertionError(f"QUBO mismatch for {bits}: polynomial={energy}, direct={direct}")
        rows.append({"bits": "".join(map(str, bits)), "energy": energy, **validation})
    return rows


def qubo_to_ising(qubo: Qubo) -> tuple[float, tuple[float, ...], dict[tuple[int, int], float]]:
    """Return c,h,J for E=c + sum h_i z_i + sum J_ij z_i z_j, x=(1-z)/2."""
    c = qubo.offset + sum(a / 2 for a in qubo.linear) + sum(v / 4 for v in qubo.quadratic.values())
    h = [ -a / 2 for a in qubo.linear ]
    j = {pair: value / 4 for pair, value in qubo.quadratic.items()}
    for (i, k), value in qubo.quadratic.items():
        h[i] -= value / 4
        h[k] -= value / 4
    return c, tuple(h), j


def ising_energy(qubo: Qubo, bits: Sequence[int]) -> float:
    c, h, j = qubo_to_ising(qubo)
    z = [1 - 2 * int(bit) for bit in bits]
    return c + sum(a * x for a, x in zip(h, z)) + sum(v * z[i] * z[k] for (i, k), v in j.items())
