"""
tests/test_inspection_and_probes.py

Deterministic test suite for environment inspection and SDK capability probes.
Uses Python unittest standard library so it runs in any standard Python interpreter.
Includes deep mock verification for simulated SDK execution, property handling,
credential isolation, and error handling paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.inspect_environment import (
    discover_python_interpreters,
    format_markdown,
    get_cpu_info,
    get_disk_info,
    get_gpu_info,
    get_installed_packages,
    get_memory_info,
    get_os_info,
    run_inspection,
)
from tools.sdk_probes.probe_ortools import probe_ortools
from tools.sdk_probes.probe_qpiai_opt import probe_opt
from tools.sdk_probes.probe_qpiai_quantum import probe_quantum
from tools.sdk_probes.run_all_probes import format_table, run_all


class TestEnvironmentInspector(unittest.TestCase):
    def test_os_info(self):
        info = get_os_info()
        self.assertIn("system", info)
        self.assertIn("release", info)
        self.assertIn("architecture", info)
        self.assertIn("python_build", info)

    def test_cpu_info(self):
        info = get_cpu_info()
        self.assertIn("logical_cores", info)
        self.assertGreaterEqual(info["logical_cores"], 1)
        self.assertIn("model", info)

    def test_memory_info(self):
        info = get_memory_info()
        self.assertIn("total_gb", info)
        self.assertIn("available_gb", info)
        if info["total_gb"] is not None:
            self.assertGreater(info["total_gb"], 0.0)

    def test_disk_info_and_error_handling(self):
        # Valid directory
        info = get_disk_info(".")
        self.assertIn("total_gb", info)
        self.assertIn("free_gb", info)

        # Invalid directory error handling
        bad_info = get_disk_info("nonexistent_dir_debugleaf_probe_xyz123")
        self.assertIn("error", bad_info)

    def test_python_discovery(self):
        interpreters = discover_python_interpreters()
        self.assertIsInstance(interpreters, list)
        self.assertGreater(len(interpreters), 0)
        paths = [i["path"].lower() for i in interpreters]
        resolved_current = str(Path(sys.executable).resolve()).lower()
        self.assertIn(resolved_current, paths)

    def test_gpu_info_no_crash(self):
        gpu = get_gpu_info()
        self.assertIn("available", gpu)
        # Verify no credentials leaked
        for key in gpu.keys():
            self.assertNotIn("secret", key.lower())
            self.assertNotIn("token", key.lower())

    def test_installed_packages(self):
        pkgs = get_installed_packages(["sys", "unittest"])
        self.assertIsInstance(pkgs, dict)

    def test_full_inspection_and_markdown(self):
        data = run_inspection(".")
        md = format_markdown(data)
        self.assertIn("# DebugLeaf Environment Inspection Report", md)
        self.assertIn("Operating System", md)
        self.assertIn("Compute & Memory", md)

    def test_format_markdown_missing_or_error_fields(self):
        mock_data = {
            "os": {
                "system": "TestOS",
                "release": "1.0",
                "version": "1.0.0",
                "architecture": "64bit",
                "machine": "x86_64",
                "python_build": "3.12.0",
            },
            "cpu": {"model": "TestCPU", "logical_cores": 4},
            "memory": {"total_gb": None, "available_gb": None, "used_percent": None},
            "disk": {"error": "Access denied"},
            "gpu": {"available": False, "note": "No GPU detected"},
            "python_interpreters": [],
            "packages": {"test_pkg": {"installed": False}},
        }
        md = format_markdown(mock_data)
        self.assertIn("Total RAM:** Unavailable", md)
        self.assertIn("Inspection error (Access denied)", md)


class TestSDKProbes(unittest.TestCase):
    def test_probe_quantum_structure(self):
        res = probe_quantum()
        self.assertEqual(res["sdk"], "qpiai-quantum")
        self.assertIn("stage_1_import", res)
        self.assertIn("stage_2_construction", res)
        self.assertIn("stage_3_local_execution", res)
        self.assertIn("stage_4_cloud_status", res)
        self.assertIn("action", res["stage_4_cloud_status"])

    def test_probe_opt_structure(self):
        res = probe_opt()
        self.assertEqual(res["sdk"], "qpiai-opt")
        self.assertIn("stage_1_import", res)
        self.assertIn("stage_2_construction", res)
        self.assertIn("stage_3_local_execution", res)
        self.assertIn("stage_4_cloud_status", res)
        self.assertIn("action", res["stage_4_cloud_status"])

    def test_probe_ortools_structure(self):
        res = probe_ortools()
        self.assertEqual(res["sdk"], "ortools")
        self.assertIn("stage_1_import", res)
        self.assertIn("stage_2_construction", res)
        self.assertIn("stage_3_local_execution", res)
        self.assertIn("stage_4_cloud_status", res)

    def test_run_all_probes_and_formatting(self):
        results = run_all()
        self.assertIn("qpiai_quantum", results)
        self.assertIn("qpiai_opt", results)
        self.assertIn("ortools", results)

        table = format_table(results)
        self.assertIn("# DebugLeaf SDK Capability Scorecard", table)
        self.assertIn("qpiai-quantum", table)
        self.assertIn("qpiai-opt", table)
        self.assertIn("ortools", table)

    def test_skip_notes_on_uninstalled(self):
        res = probe_quantum()
        if not res["stage_1_import"]["success"]:
            self.assertIn("Skipped:", res["stage_2_construction"].get("note", ""))
            self.assertIn("Skipped:", res["stage_3_local_execution"].get("note", ""))

    def test_credential_isolation(self):
        test_qcloud_key = "SECRET_QCLOUD_KEY_PROBE_TEST_99"
        test_opt_key = "SECRET_OPT_KEY_PROBE_TEST_88"

        with patch.dict(os.environ, {"QPIAI_QCLOUD_API_KEY": test_qcloud_key, "QPIAI_OPT_API_KEY": test_opt_key}):
            res_q = probe_quantum()
            res_opt = probe_opt()
            all_res = run_all()
            table_md = format_table(all_res)

            q_json = json.dumps(res_q)
            opt_json = json.dumps(res_opt)
            all_json = json.dumps(all_res)

            # Assert actual secret values are strictly excluded from output
            self.assertNotIn(test_qcloud_key, q_json)
            self.assertNotIn(test_opt_key, opt_json)
            self.assertNotIn(test_qcloud_key, all_json)
            self.assertNotIn(test_opt_key, all_json)
            self.assertNotIn(test_qcloud_key, table_md)
            self.assertNotIn(test_opt_key, table_md)

            # Assert boolean configured flag is True
            self.assertTrue(res_q["stage_4_cloud_status"]["credentials_configured"])
            self.assertTrue(res_opt["stage_4_cloud_status"]["credentials_configured"])

    def test_probe_quantum_mocked_execution_and_property_access(self):
        """
        Simulate successful qpiai_quantum import and execution.
        Critically tests that StatevectorSimulator.name as a @property (not a method)
        does not cause a TypeError, and that circuit construction succeeds.
        """
        mock_qpiai_quantum = MagicMock()
        mock_qpiai_quantum.__version__ = "0.7.5"

        class MockCircuit:
            def __init__(self, n, name="ProbeBellState"):
                self.num_qubits = n
                self.name = name

            def h(self, q):
                pass

            def cx(self, q1, q2):
                pass

            def measure_all(self):
                pass

        class MockSimulatorResult:
            @property
            def counts(self):
                return {"00": 52, "11": 48}

        class MockStatevectorSimulator:
            @property
            def name(self) -> str:
                return "QpiAI-QSV-Local"

            def run(self, circuit, shots=100, seed=42):
                return MockSimulatorResult()

        mock_qpiai_quantum.Circuit = MockCircuit
        mock_qpiai_quantum.QpiAIQuantumAuth = MagicMock()
        mock_simulator_mod = MagicMock()
        mock_simulator_mod.StatevectorSimulator = MockStatevectorSimulator

        with patch.dict(
            sys.modules,
            {
                "qpiai_quantum": mock_qpiai_quantum,
                "qpiai_quantum.simulator": mock_simulator_mod,
            },
        ):
            res = probe_quantum()
            self.assertTrue(res["stage_1_import"]["success"])
            self.assertTrue(res["stage_2_construction"]["success"])
            self.assertEqual(res["stage_2_construction"]["qubits"], 2)
            self.assertTrue(res["stage_3_local_execution"]["success"])
            self.assertEqual(res["stage_3_local_execution"]["simulator_type"], "QpiAI-QSV-Local")
            self.assertEqual(res["stage_3_local_execution"]["counts"], {"00": 52, "11": 48})
            self.assertTrue(res["stage_3_local_execution"]["correlated_superposition_verified"])

    def test_probe_opt_mocked_execution(self):
        """Simulate successful qpiai_opt import, QUBO compilation, and circuit building."""
        mock_qpiai_opt = MagicMock()
        mock_qpiai_opt.__version__ = "0.0.9"

        class MockModel:
            def __init__(self, name):
                self.name = name

            def minimize(self, expr):
                pass

            def to_qubo_dict(self, matrix_format="upper"):
                return {(0, 0): 2.0, (1, 1): 3.0, (0, 1): -4.0}, 0.0

        class MockQUBOProblem:
            num_variables = 2
            matrix_format = "symmetric"

            @classmethod
            def from_model(cls, m):
                return cls()

        mock_modeller = MagicMock()
        mock_modeller.Model = MockModel
        mock_modeller.variables.Binary = MagicMock(return_value=1)
        mock_modeller.variables.Variable = MagicMock()

        mock_models = MagicMock()
        mock_models.QUBOProblem = MockQUBOProblem

        mock_quantum = MagicMock()
        mock_quantum.build_circuit = MagicMock(return_value=MagicMock(__class__=type("MockQCircuit", (), {})))
        mock_quantum.QAOA = MagicMock()
        mock_quantum.WarmStart = MagicMock()

        mock_solvers_client = MagicMock()
        mock_solvers_client.Client = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "qpiai_opt": mock_qpiai_opt,
                "qpiai_opt.opt_modeller": mock_modeller,
                "qpiai_opt.opt_modeller.variables": mock_modeller.variables,
                "qpiai_opt.opt_solvers.qubo.models": mock_models,
                "qpiai_opt.opt_solvers.client": mock_solvers_client,
                "qpiai_opt.quantum": mock_quantum,
            },
        ):
            with patch.dict(os.environ, {}, clear=True):
                res = probe_opt()
                self.assertTrue(res["stage_1_import"]["success"])
                self.assertTrue(res["stage_2_construction"]["success"])
                self.assertEqual(res["stage_2_construction"]["num_variables"], 2)
                self.assertFalse(res["stage_3_local_execution"]["success"])
                self.assertEqual(res["stage_3_local_execution"]["status"], "REQUIRES_OPT_KEY")


class TestCLIExecution(unittest.TestCase):
    def test_cli_inspect_environment_json(self):
        cmd = [sys.executable, str(root_dir / "tools" / "inspect_environment.py"), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIn("os", data)
        self.assertIn("cpu", data)
        self.assertIn("memory", data)
        self.assertIn("python_interpreters", data)

    def test_cli_run_all_probes_json(self):
        cmd = [sys.executable, str(root_dir / "tools" / "sdk_probes" / "run_all_probes.py"), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIn("qpiai_quantum", data)
        self.assertIn("qpiai_opt", data)
        self.assertIn("ortools", data)


if __name__ == "__main__":
    unittest.main()
