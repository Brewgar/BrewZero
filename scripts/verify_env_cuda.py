"""Gate 0 environment verification script.

Writes the report to both stdout and a file so that shell-integration output
truncation cannot lose the result.
"""
import os
import sys

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "environment_report.txt")

lines = []
try:
    import torch
except Exception as exc:  # noqa: BLE001
    lines.append(f"torch import FAILED: {exc}")
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write(report)
    print(report)
    sys.exit(1)

lines.append("torch import: OK")
lines.append(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    lines.append(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    props = torch.cuda.get_device_properties(0)
    lines.append(f"GPU VRAM (MiB): {props.total_memory / 2 ** 20:.1f}")
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    lines.append(f"CUDA matmul sanity: {y > 0}")
else:
    lines.append("CUDA UNAVAILABLE -> STOP per spec")
lines.append(f"PyTorch: {torch.__version__}")
lines.append(f"CUDA runtime (torch.version.cuda): {torch.version.cuda}")
lines.append(f"CUDNN: {torch.backends.cudnn.version()}")

report = "\n".join(lines)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w") as fh:
    fh.write(report)
print(report)