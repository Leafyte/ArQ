"""One frozen-protocol prospective scenario evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.prospective_dispatch import run_prospective_study  # noqa: E402

EXPECTED_PROTOCOL_SHA256 = "60CDC7A0B6E676B2BE9FFA7928BC0BE505BBF9870B9D65CF48D833F935AD86F8"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_bytes = (root / "experiments" / "protocol" / "prospective-dispatch-vqc-v1.json").read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest().upper()
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol hash mismatch: {protocol_hash}")
    protocol = json.loads(protocol_bytes)
    fixture_path = root / protocol["source_fixture"]["path"]
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest().upper()
    if fixture_hash != protocol["source_fixture"]["sha256_file_bytes"]:
        raise RuntimeError("source fixture hash differs from frozen protocol")
    scenario = json.loads(fixture_path.read_text(encoding="utf-8"))
    result = run_prospective_study(scenario, protocol_hash)
    result["runtime"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                         "execution": "local CPU exact statevector; paired hidden realized scenarios; no external calls"}
    output = root / "experiments" / "artifacts" / "prospective-dispatch-results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({"protocol_sha256": protocol_hash,
                      "gradient_error": result["training"]["decision_gradient_max_absolute_error"],
                      "test_metrics": result["test_metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
