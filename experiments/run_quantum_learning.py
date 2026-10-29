"""Run the frozen tiny synthetic VQC study and save its raw JSON results."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.quantum_learning import run_study  # noqa: E402


def main() -> None:
    payload = run_study()
    payload["runtime"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "execution": "local CPU exact statevector; no vendor SDK invocation",
    }
    out = Path(__file__).resolve().parent / "artifacts" / "quantum-learning-results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(payload["gradient_check"], indent=2))
    print(json.dumps(payload["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
