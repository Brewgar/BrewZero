"""Raw UCI probe: dump engine output for the fool's-mate position."""
import subprocess

SF = r"stockfish\stockfish-windows-x86-64-avx2.exe"
cmds = [
    "uci",
    "setoption name Threads value 1",
    "setoption name Hash value 32",
    "setoption name UCI_ShowWDL value true",
    "position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "go depth 12",
]
proc = subprocess.Popen(
    [SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
)
assert proc.stdin is not None and proc.stdout is not None
for c in cmds:
    proc.stdin.write(c + "\n")
proc.stdin.flush()
lines = []
for line in proc.stdout:
    lines.append(line.rstrip())
    if line.startswith("bestmove"):
        break
proc.kill()
for ln in lines[-14:]:
    print(ln)