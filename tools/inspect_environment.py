#!/usr/bin/env python3
"""
tools/inspect_environment.py

Lightweight, read-only environment inspection tool for DebugLeaf.
Portable across Windows, Linux, and macOS.
Reports OS, CPU, RAM, disk, available Python interpreters, GPU (via installed tools only),
and installed target packages.

SAFETY RULES:
- Never dumps environment variables or credential values.
- Never performs installations, driver updates, or system modifications.
- Runs with zero external dependencies (pure Python standard library).
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_os_info() -> Dict[str, Any]:
    """Inspect operating system and system architecture."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor() or "Unknown",
        "python_build": platform.python_version(),
        "platform_string": platform.platform(),
    }


def get_cpu_info() -> Dict[str, Any]:
    """Inspect CPU information without installing external tools."""
    logical_cores = os.cpu_count() or 1
    cpu_model = platform.processor() or "Unknown"

    if platform.system() == "Windows":
        try:
            # Query CPU name via PowerShell Get-CimInstance without touching environment
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            )
            out = res.stdout.strip()
            if out:
                cpu_model = out
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    lower = line.lower()
                    if lower.startswith("model name") or lower.startswith("hardware") or lower.startswith("model\t"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return {
        "model": cpu_model,
        "logical_cores": logical_cores,
    }


def get_memory_info() -> Dict[str, Any]:
    """Inspect physical RAM using native OS facilities without external packages."""
    total_bytes = 0
    available_bytes = 0

    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_bytes = int(stat.ullTotalPhys)
                available_bytes = int(stat.ullAvailPhys)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
                if "MemTotal" in meminfo:
                    total_bytes = int(meminfo["MemTotal"].split()[0]) * 1024
                if "MemAvailable" in meminfo:
                    available_bytes = int(meminfo["MemAvailable"].split()[0]) * 1024
        except Exception:
            pass

    gb = 1024**3
    total_gb = round(total_bytes / gb, 2) if total_bytes else None
    available_gb = round(available_bytes / gb, 2) if available_bytes else None
    used_percent = (
        round(100.0 * (1.0 - available_bytes / total_bytes), 1)
        if total_bytes and available_bytes
        else None
    )

    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "used_percent": used_percent,
    }


def get_disk_info(target_dir: str = ".") -> Dict[str, Any]:
    """Inspect disk space for the target directory drive."""
    try:
        total, used, free = shutil.disk_usage(target_dir)
        gb = 1024**3
        return {
            "path": str(Path(target_dir).resolve()),
            "total_gb": round(total / gb, 2),
            "free_gb": round(free / gb, 2),
            "used_gb": round(used / gb, 2),
            "used_percent": round(100.0 * used / total, 1) if total else None,
        }
    except Exception as exc:
        return {"error": str(exc)}


def discover_python_interpreters() -> List[Dict[str, str]]:
    """Discover available Python executables on the host."""
    interpreters: List[Dict[str, str]] = []
    seen_paths = set()

    current_exe = sys.executable
    if current_exe:
        resolved = str(Path(current_exe).resolve())
        seen_paths.add(resolved.lower())
        interpreters.append(
            {
                "path": resolved,
                "version": platform.python_version(),
                "status": "active_interpreter",
            }
        )

    # Candidate command names
    candidate_names = [
        "python",
        "python3",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.13",
        "python3.14",
        "py",
    ]

    for name in candidate_names:
        cmd_path = shutil.which(name)
        if cmd_path:
            try:
                resolved = str(Path(cmd_path).resolve())
                if resolved.lower() not in seen_paths:
                    seen_paths.add(resolved.lower())
                    res = subprocess.run(
                        [resolved, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    ver = (res.stdout or res.stderr).strip()
                    interpreters.append(
                        {
                            "path": resolved,
                            "version": ver.replace("Python ", "") if ver else "Unknown",
                            "status": "system_path",
                        }
                    )
            except Exception:
                pass

    # Check project-local virtual environments
    project_root = Path(__file__).resolve().parent.parent
    venv_candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
        project_root / "venv" / "Scripts" / "python.exe",
        project_root / "venv" / "bin" / "python",
    ]
    for venv_py in venv_candidates:
        if venv_py.is_file():
            try:
                resolved = str(venv_py.resolve())
                if resolved.lower() not in seen_paths:
                    seen_paths.add(resolved.lower())
                    res = subprocess.run(
                        [resolved, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    ver = (res.stdout or res.stderr).strip()
                    interpreters.append(
                        {
                            "path": resolved,
                            "version": ver.replace("Python ", "") if ver else "Unknown",
                            "status": "project_venv",
                        }
                    )
            except Exception:
                pass

    # Check uv python list if uv is installed
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            res = subprocess.run(
                [uv_path, "python", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in res.stdout.splitlines():
                line = line.strip()
                if not line or "<download available>" in line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and Path(parts[1]).is_file():
                    p = str(Path(parts[1]).resolve())
                    if p.lower() not in seen_paths:
                        seen_paths.add(p.lower())
                        interpreters.append(
                            {
                                "path": p,
                                "version": parts[0],
                                "status": "uv_managed",
                            }
                        )
        except Exception:
            pass

    return interpreters


def get_gpu_info() -> Dict[str, Any]:
    """
    Check GPU presence strictly through already-installed read-only tools.
    Never attempts driver install or modification.
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    # Also check common standard paths on Windows and Linux
    if not nvidia_smi_path:
        if platform.system() == "Windows":
            candidates = [
                Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
                Path(r"C:\Windows\System32\nvidia-smi.exe"),
            ]
        else:
            candidates = [
                Path("/usr/bin/nvidia-smi"),
                Path("/usr/local/cuda/bin/nvidia-smi"),
            ]
        for c in candidates:
            if c.is_file():
                nvidia_smi_path = str(c)
                break

    if nvidia_smi_path:
        try:
            cmd = [
                nvidia_smi_path,
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            )
            if res.returncode == 0 and res.stdout.strip():
                gpus = []
                for line in res.stdout.strip().splitlines():
                    fields = [f.strip() for f in line.split(",")]
                    if len(fields) >= 4:
                        gpus.append(
                            {
                                "model": fields[0],
                                "driver_version": fields[1],
                                "total_memory_mb": float(fields[2]),
                                "free_memory_mb": float(fields[3]),
                            }
                        )
                return {
                    "tool": "nvidia-smi",
                    "available": True,
                    "count": len(gpus),
                    "devices": gpus,
                }
            elif res.returncode != 0:
                err_msg = (res.stderr or res.stdout).strip() or f"nvidia-smi exited with code {res.returncode}"
                return {
                    "tool": "nvidia-smi",
                    "available": False,
                    "error": err_msg,
                    "note": f"nvidia-smi detected at {nvidia_smi_path} but failed execution: {err_msg}",
                }
        except Exception as exc:
            return {"tool": "nvidia-smi", "available": False, "error": str(exc)}

    # Secondary inspection: Windows CIM video controllers
    if platform.system() == "Windows":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -Property Name, DriverVersion, AdapterRAM | ConvertTo-Json",
            ]
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                devices_raw = [data] if isinstance(data, dict) else data
                devices = []
                for d in devices_raw:
                    name = d.get("Name", "Unknown")
                    drv = d.get("DriverVersion", "Unknown")
                    ram = d.get("AdapterRAM")
                    ram_mb = round(ram / (1024**2), 1) if ram else None
                    devices.append(
                        {
                            "model": name,
                            "driver_version": drv,
                            "adapter_ram_mb": ram_mb,
                        }
                    )
                return {
                    "tool": "Win32_VideoController",
                    "available": True,
                    "count": len(devices),
                    "devices": devices,
                    "note": "NVIDIA SMI not detected on PATH; reporting OS display adapter.",
                }
        except Exception:
            pass

    return {
        "tool": None,
        "available": False,
        "note": "No NVIDIA GPU tool or accelerator detected via installed commands.",
    }


def get_installed_packages(
    target_packages: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Query versions of key target packages in the active Python environment."""
    if target_packages is None:
        target_packages = [
            "qpiai-quantum",
            "qpiai-opt",
            "ortools",
            "numpy",
            "scipy",
            "pydantic",
            "highspy",
            "cvxpy",
            "networkx",
            "pytest",
            "torch",
        ]

    results: Dict[str, Dict[str, Any]] = {}
    for pkg in target_packages:
        try:
            ver = importlib.metadata.version(pkg)
            results[pkg] = {"installed": True, "version": ver}
        except importlib.metadata.PackageNotFoundError:
            results[pkg] = {"installed": False, "version": None}
        except Exception as exc:
            results[pkg] = {"installed": False, "error": str(exc)}

    return results


def run_inspection(workspace_dir: str = ".") -> Dict[str, Any]:
    """Execute complete environment inspection safely."""
    return {
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(workspace_dir),
        "gpu": get_gpu_info(),
        "python_interpreters": discover_python_interpreters(),
        "packages": get_installed_packages(),
    }


def format_markdown(data: Dict[str, Any]) -> str:
    """Format inspection data into readable Markdown."""
    mem = data.get("memory", {})
    if mem.get("total_gb") is not None:
        mem_str = f"{mem['total_gb']} GB (Available: {mem.get('available_gb', 'N/A')} GB, Used: {mem.get('used_percent', 'N/A')}%)"
    else:
        mem_str = "Unavailable"

    disk = data.get("disk", {})
    if "error" in disk:
        disk_str = f"Inspection error ({disk['error']})"
    elif disk.get("total_gb") is not None:
        disk_str = f"{disk['total_gb']} GB Total, {disk.get('free_gb')} GB Free ({disk.get('path')})"
    else:
        disk_str = "Unavailable"

    lines: List[str] = [
        "# DebugLeaf Environment Inspection Report",
        "",
        "## 1. Operating System",
        f"- **OS:** {data['os']['system']} {data['os']['release']} ({data['os']['version']})",
        f"- **Architecture:** {data['os']['architecture']} ({data['os']['machine']})",
        f"- **Active Python:** {data['os']['python_build']}",
        "",
        "## 2. Compute & Memory",
        f"- **CPU Model:** {data['cpu']['model']}",
        f"- **Logical Cores:** {data['cpu']['logical_cores']}",
        f"- **Total RAM:** {mem_str}",
        f"- **Workspace Disk:** {disk_str}",
        "",
        "## 3. GPU / Accelerator Status",
    ]

    gpu = data["gpu"]
    if gpu.get("available") and gpu.get("devices"):
        for dev in gpu["devices"]:
            ram_info = f", VRAM: {dev.get('total_memory_mb') or dev.get('adapter_ram_mb')} MB"
            lines.append(
                f"- **Device:** {dev['model']} (Driver: {dev.get('driver_version')}{ram_info})"
            )
        if gpu.get("note"):
            lines.append(f"  *{gpu['note']}*")
    else:
        lines.append(f"- **Status:** {gpu.get('note', 'No GPU detected')}")

    lines.extend(["", "## 4. Discovered Python Interpreters"])
    for interp in data["python_interpreters"]:
        lines.append(
            f"- `{interp['path']}` (Version: {interp['version']}, Source: {interp['status']})"
        )

    lines.extend(["", "## 5. Active Environment Target Packages"])
    for pkg, pinfo in data["packages"].items():
        if pinfo.get("installed"):
            lines.append(f"- `{pkg}`: **{pinfo['version']}** (Installed)")
        else:
            lines.append(f"- `{pkg}`: *Not installed*")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect host environment for DebugLeaf development and Legion execution."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted markdown.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace root directory to inspect disk usage (default: .)",
    )
    args = parser.parse_args()

    report = run_inspection(workspace_dir=args.workspace)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_markdown(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
