import pathlib
import os
import argparse
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import torch
from evaluate import pit, random_player, create_mcts_player, create_shallow_player
from main import last_model_number
from mcts_py import PyHex

parser = argparse.ArgumentParser(description='Run a tournament!')
parser.add_argument('--processes', type=int, default=cpu_count() // 2,
                    help='number of processes to spin up')
parser.add_argument('--games', type=int, default=100,
                    help='number of evaluation games per model')
parser.add_argument('--mcts-iterations', type=int, default=0,
                    help='mcts iterations per move, or 0 for shallow network against random')
parser.add_argument('--dir', nargs='+', required=True,
                    help='models to evaluate')
args = parser.parse_args()

board_size = None
for dir in args.dir:
    base_path = '../runs/{}'.format(dir)
    config_path = '{}/config.json'.format(base_path)
    if not pathlib.Path(config_path).is_file():
        print('No config for {}!'.format(dir)) 
        exit()
    with open(config_path) as config_file:
        config = json.load(config_file)
    if board_size is None:
        board_size = config['board_size']
    elif config['board_size'] != board_size:
        print('Board sizes do not match!')
        exit()

players = ['baseline'] + args.dir

def load_model(player):
    if player == 'baseline':
        return create_mcts_player(None, mcts_iterations=args.mcts_iterations) if args.mcts_iterations else random_player
    base_path = '../runs/{}'.format(dir)
    models_path = '{}/models'.format(base_path)
    model_path = '{}/{}.pt'.format(models_path, last_model_number(models_path))
    net = torch.load(model_path)
    return create_mcts_player(net, mcts_iterations=args.mcts_iterations) if args.mcts_iterations else create_shallow_player(net)

for player1 in players:
    for player2 in players:
        winrate = 0
        for _ in range(args.games):
            p1, p2 = load_model(player1), load_model(player2)
            winrate += pit(PyHex(board_size), p1, p2)
        print(player1, player2, winrate)
