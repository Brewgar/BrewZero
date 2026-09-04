import ast, sys

FILES = [
    'gui/chessboard.py', 'gui/play_game.py', 'gui/worker.py', 'gui/app.py',
    'gui/state.py', 'gui/__init__.py', 'train/checkpoint.py', 'train/train.py',
    'train/config.py', 'tests/test_chessboard.py', 'tests/test_play_controller.py',
    'tests/test_checkpoint_policy.py', 'tests/test_gui.py',
]
ok = True
for f in FILES:
    try:
        ast.parse(open(f, encoding='utf-8').read(), filename=f)
        print('OK  ', f)
    except SyntaxError as exc:
        ok = False
        print('FAIL', f, '->', exc)
print('ALL-SYNTAX-OK' if ok else 'SYNTAX-ERRORS')
sys.exit(0 if ok else 1)