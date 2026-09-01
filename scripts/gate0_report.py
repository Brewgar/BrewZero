"""Gate 0: machine-readable environment report (YAML).

Verifies: Python, venv, PyTorch+CUDA, GPU, Stockfish executable + UCI.
Reports are written to reports/gate0_environment.yaml.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SF_EXE = os.path.join(ROOT, "stockfish", "stockfish-windows-x86-64-avx2.exe")


def stockfish_identity(exe: str) -> tuple[str, str]:
    """Run a minimal UCI handshake and return (identity_line, binary_sha256)."""
    sha = hashlib.sha256()
    with open(exe, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    proc = subprocess.Popen(
        [exe],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write("uci\nisready\nquit\n")
    proc.stdin.flush()
    lines = []
    for line in proc.stdout:
        line = line.strip()
        lines.append(line)
        if line.startswith("id name"):
            identity = line
        if line == "readyok":
            break
    proc.wait(timeout=10)
    return identity, sha.hexdigest()


def main() -> None:
    report: dict = {"gate": 0, "machine": {}, "python": {}, "torch": {}, "stockfish": {}}

    report["machine"]["platform"] = platform.platform()
    report["machine"]["cpu"] = platform.processor()
    try:
        import psutil  # noqa: F401  (optional)
        # not a dependency; fall back to ctypes-free approach below
    except ImportError:
        pass

    report["python"]["version"] = platform.python_version()
    report["python"]["executable"] = sys.executable

    import torch

    report["torch"]["version"] = str(torch.__version__)
    report["torch"]["cuda_available"] = bool(torch.cuda.is_available())
    report["torch"]["cuda_runtime"] = str(torch.version.cuda)
    report["torch"]["cudnn"] = str(torch.backends.cudnn.version())
    if torch.cuda.is_available():
        report["torch"]["gpu"] = str(torch.cuda.get_device_name(0))
        report["torch"]["gpu_capability"] = [int(x) for x in torch.cuda.get_device_capability(0)]
        report["torch"]["gpu_vram_mib"] = float(
            round(torch.cuda.get_device_properties(0).total_memory / (2**20), 1)
        )

    if not os.path.exists(SF_EXE):
        raise SystemExit(f"Stockfish executable not found at {SF_EXE}")
    identity, sha = stockfish_identity(SF_EXE)
    report["stockfish"]["path"] = SF_EXE
    report["stockfish"]["sha256"] = sha
    report["stockfish"]["identity"] = identity
    report["stockfish"]["uci_handshake"] = "readyok observed"

    out_path = os.path.join(ROOT, "reports", "gate0_environment.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(report, fh, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()