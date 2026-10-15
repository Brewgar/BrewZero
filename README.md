# BrewZero

A chess AI that learns by playing with itself (pause), written entirely in Python.

A neural network plays both sides of the board, loses to itself half the time, wins against itself the other half, and slowly figures out what it's doing. Stockfish 18 sits on the sidelines during training and coaches it, so the model learns from how good its moves were and not only from who won in the end.

## Read this first

This is a proof of concept and a fun little machine learning project. It is not trying to replace a real chess engine, and it will not beat one. Also, yes, the "Zero" in the name is very misleading. I'm aware.

## What's under the hood

- **PyTorch** for the neural network and GPU training
- **PPO + GAE** for the reinforcement learning
- **python-chess** for, well, chess
- **Stockfish 18** for evaluation and feedback

The model is a residual network with separate heads for policy, game value, training value, and an optional Stockfish value. Self-play, training, evaluation, checkpointing, and human play all go through the same APIs.

## Installation

You'll need Python 3.14 and a virtual environment.

1. Create and activate your venv.
2. Install the dependencies: `pip install -r requirements.txt`
3. Install the CUDA build of PyTorch: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`
4. Download the Stockfish 18 executable and drop it into the `stockfish/` folder. It isn't included in the repo.

Check that your setup works with `python scripts/gate0_report.py`, then run the tests with `pytest tests/ -q`.

## Training and evaluation

There are several training configs so you can compare different ways of teaching the model. The terminal-only baseline looks like this:

```
python main.py --config configs/terminal_only.yaml --hours 4
```

The Stockfish-dense, combined, and auxiliary-value experiments each have their own YAML in `configs/`.

Add `--resume` to continue a run. To evaluate a checkpoint, use `--eval-only --checkpoint ...`.

Evaluation looks at real games against fixed opponents: wins, draws, losses, score, and estimated Elo. A training curve going up is not proof the model improved, so it isn't treated as proof here.

## GUI

There's a lightweight Tkinter GUI for training controls, loading checkpoints, running evaluations, and viewing model stats. Launch it with `python gui.py`. It uses the same APIs as the command line, so it doesn't have its own secret RL logic.

## Tests

More than 165 unit and integration tests cover the code. GUI tests run with `pytest tests/test_gui.py -v`. There's also a small smoke config for checking that the whole pipeline works before you commit to a long run and find out at hour three that it didn't.

## License

MIT.
