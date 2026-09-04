"""Sample GPU/CPU/RAM during a benchmark run (Windows)."""
import subprocess
import sys
import time


def main(duration_s: float, out: str) -> None:
    deadline = time.time() + duration_s
    with open(out, "w") as fh:
        fh.write("elapsed,gpu_util_pct,vram_mib\n")
        while time.time() < deadline:
            t = time.time()
            try:
                out_smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                util, vram = out_smi.split(",")
                fh.write(f"{t:.1f},{int(util)},{int(vram)}\n")
            except Exception:
                fh.write(f"{t:.1f},N/A,N/A\n")
            fh.flush()
            time.sleep(2.0)


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    out = sys.argv[2] if len(sys.argv) > 2 else "gpu_util.log"
    main(duration, out)