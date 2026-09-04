# BrewZero

## Purpose

BrewZero is a self-play reinforcement-learning chess system that trains a neural network to play both sides of a chess game. During training, Stockfish 18 provides auxiliary feedback such as WDL-based evaluation, expected score, and regret, helping the model learn from the quality of its decisions in addition to the final game result. This project is made to be just a proof of concept. BrewZero doesn't claim or aim to become a replacement for a strong chess engine. Its rather just a fun little machine learning project. And yes, the Zero in the model's name is very misleading.

## Technologies

The project is built in entirely in Python and uses PyTorch for the neural network and GPU accelerated training, PPO and GAE for reinforcement learning, python-chess for well, chess. And Stockfish 18 for evaluation and feedback. The model uses a residual neural-network architecture with separate policy, game-value, training-value, and optional Stockfish-value components. Training, self-play, evaluation, checkpointing, and human play are connected through the same underlying APIs.

## Installation

BrewZero uses a Python vem and requires Python 3.14. After creating and running the environment, install the project dependencies via requirements.txt and install the CUDA-enabled PyTorch build with `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`. The Stockfish executable is configured separately rather than being included in the project. The environment can be checked with `python scripts/gate0_report.py`, and the complete test suite can be run with `pytest tests/ -q`. Everything you need should be included in the github repository, except for the Stockfish 18 Executable file. Download and drop it into /stockfish.

## Training and Evaluation

The project provides multiple training configurations to compare different learning approaches. The terminal only baseline can be started with `python main.py --config configs/terminal_only.yaml --hours 4`, while Stockfish-dense, combined, and auxiliary-value experiments use their corresponding YAML configurations. Training can be resumed with `--resume`, and checkpoints can be evaluated using `--eval-only --checkpoint ...`. Evaluation focuses on actual game performance using wins, draws, losses, score, and estimated Elo against fixed opponents rather than treating training statistics as proof of improvement.

## GUI

BrewZero includes a lightweight Tkinter gui that provides a simple interface for controlling and monitoring the entire system. The gui can be launched with `python gui.py` and provides training controls, checkpoint loading, evaluation, model statistics etc. The gui uses the same training and evaluation APIs as the commandline interface instead of implementing separate rl logic, allowing training, evaluation, and human play to remain consistent with the system.

## Development and Testing

More than 165 unit and integration tests cover the functionality, with additional GUI tests available through `pytest tests/test_gui.py -v`. A small smoke configuration is provided to verify the complete training pipeline before running longer experiments. Checkpoints and training logs are stored separately so that experiments can be resumed or evaluated later.
