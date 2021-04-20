# Learning hex from self play

## Dependencies and building the extension

The project includes a pytorch model and a python extension written in Rust for the game evaluation and tree search implementation.

So you'll need python and the dependencies in `py/requirements.txt`. As well as rust (unfortunately, you need the nightly channel).

See `install.sh` for how these would be installed on an Ubuntu machine. (In `mcts`, run `cargo build --release`. On linux, you'll need to copy the library for python to see it: `cd py && ln ../mcts/target/release/libmcts_py.so mcts_py.so`.)

## Training

Setup your desired parameters in `config.py`. Checkpoints are saved in the `runs` directory based on the config.

See `python main.py --help` for training options.

Self play, training, and evaluation run in parallel and on multiple processes. Training times on a recent MacBook with 12 processes range from ~1sec/game on a 5x5 board with 1000 MCTS iterations, to over 1min/game on larger boards with more iterations.

## Evaluation and plotting

`evaluate.py` runs agents against baseline models (baselines are random, and MCTS with random rollout), saving results next to the trained models.

`plot.py` generates the plots used in the paper, based on the results from these evaluations.

You can also play against the trained model with a barest-bones terminal interface, with `human_play.py`.
