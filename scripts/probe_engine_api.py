"""Probe python-chess engine API surface (WDL keys, score types)."""
import chess
import chess.engine

SF = r"stockfish\stockfish-windows-x86-64-avx2.exe"

engine = chess.engine.SimpleEngine.popen_uci(SF)
engine.configure({"Threads": 1, "Hash": 64})
# Try to enable WDL output (Stockfish 14+)
try:
    engine.configure({"UCI_ShowWDL": "true"})
    print("UCI_ShowWDL set OK")
except Exception as exc:  # noqa: BLE001
    print("UCI_ShowWDL failed:", exc)

board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
info = engine.analyse(board, chess.engine.Limit(depth=10))

print("info keys:", sorted(info.keys()))
score = info["score"]
print("score type:", type(score))
print("score.white():", score.white(), "type:", type(score.white()))
print("score.white().is_mate():", score.white().is_mate(), "mate:", score.white().mate, "cp:", score.white().cp)
print("pv[0]:", info["pv"][0])
if "wdl" in info:
    w = info["wdl"]
    print("wdl type:", type(w))
    print("wdl repr:", w)
    for attr in ("pov", "wins", "draws", "losses", "expectation"):
        print("  has", attr, ":", hasattr(w, attr))
# deeper
info2 = engine.analyse(board, chess.engine.Limit(depth=12), game=board.copy(stack=False))
print("depth keys on info2:", sorted(info2.keys()))
# check WDL for a mate position
engine.configure({"UCI_ShowWDL": "true"})
mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
info3 = engine.analyse(mate, chess.engine.Limit(depth=10))
print("mate pos score:", info3["score"].white(), "wdl present:", "wdl" in info3)
engine.quit()
print("probe done")