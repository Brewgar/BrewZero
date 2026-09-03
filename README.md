# BrewZero

## Purpose

BrewZero is a self-play reinforcement-learning chess system that trains a neural network to play both sides of a chess game. During training, Stockfish 18 provides auxiliary feedback such as WDL-based evaluation, expected score, and regret, helping the model learn from the quality of its decisions in addition to the final game result. The main purpose of the project is just a proof of concept. BrewZero doesn't aim to become a replacement for a strong chess engine. Its rather just a fun little machine learning project. And yes, the Zero in the model's name is very misleading.

## Technologies

The project is built in Python and uses PyTorch for the neural network and GPU-accelerated training, PPO and GAE for reinforcement learning, python-chess for chess rules and legal move handling, and Stockfish 18 for evaluation and feedback. The model uses a residual neural-network architecture with separate policy, game-value, training-value, and optional Stockfish-value components. Training, self-play, evaluation, checkpointing, and human play are connected through the same underlying APIs.

## Installation

BrewZero uses a Python virtual environment and requires Python 3.14. After creating and activating the environment, install the project dependencies with requirements.txt and install the CUDA-enabled PyTorch build with `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`. The Stockfish executable is configured separately rather than being hard-coded into the project. The environment can be checked with `python scripts/gate0_report.py`, and the complete test suite can be run with `pytest tests/ -q`. Everything you need should be included in the github repository, except for the Stockfish 18 Executable file. Download and drop it into /stockfish.

## Training and Evaluation

The project provides several training configurations for comparing different learning approaches. The terminal-only baseline can be started with `python main.py --config configs/terminal_only.yaml --hours 4`, while Stockfish-dense, combined, and auxiliary-value experiments use their corresponding YAML configurations. Training can be resumed with `--resume`, and checkpoints can be evaluated using `--eval-only --checkpoint ...`. Evaluation focuses on actual game performance using wins, draws, losses, score, and estimated Elo against fixed opponents rather than treating training statistics as proof of improvement.

## GUI

BrewZero also includes a lightweight Tkinter GUI that provides a simple interface for controlling and monitoring the entire system. It can be launched with `python gui.py` and provides training controls, checkpoint loading, evaluation, model statistics, an event log, and a Play Against AI function. The GUI uses the same training and evaluation APIs as the command-line interface instead of implementing separate RL logic, allowing training, evaluation, and human play to remain consistent with the underlying system.

## Development and Testing

The project is organized into separate components for the chess environment, Stockfish integration, model, self-play, training, evaluation, benchmarking, GUI, and testing. More than 165 unit and integration tests cover the core functionality, with additional GUI tests available through `pytest tests/test_gui.py -v`. A small smoke configuration is also provided to verify the complete training pipeline before running longer experiments. Checkpoints and training logs are stored separately so that experiments can be resumed and evaluated later.
