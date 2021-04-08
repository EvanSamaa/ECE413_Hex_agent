import pathlib
import os

import torch
from evaluate import create_mcts_player, create_shallow_player, human_player, pit
from mcts_py import PyHex
from config import config

import argparse

parser = argparse.ArgumentParser(description='Play against a hex agent')
parser.add_argument('--shallow', action='store_true',
    help='Play directly against the policy network (no search)')
args = parser.parse_args()

models_path = '../runs/{}/models'.format(config['directory'])

pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
last_model = max(int(os.path.splitext(f)[0]) for f in os.listdir(models_path)) if os.listdir(models_path) else 0
model_path = '{}/{}.pt'.format(models_path, last_model)

net = torch.load(model_path)
print('Loaded model from "{}"'.format(model_path))

player1 = create_shallow_player(net) if args.shallow else create_mcts_player(net, mcts_iterations=config['mcts_iterations'])
player2 = human_player
pit(PyHex(config['board_size']), player1, player2, verbose=True)

