"""Run candidate-route decision-focused VQC study on the frozen tiny dispatch fixture."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.learning_dispatch import run_dispatch_study  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scenario = json.loads((root / "fixtures" / "dispatch_tiny.json").read_text(encoding="utf-8"))
    result = run_dispatch_study(scenario)
    result["runtime"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                         "execution": "local CPU exact statevector; validator recomputes selected plans"}
    out = Path(__file__).resolve().parent / "artifacts" / "dispatch-vqc-results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(result["model_training"], indent=2, sort_keys=True))
    print(json.dumps(result["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
