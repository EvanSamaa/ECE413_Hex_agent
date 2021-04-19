import random
from mcts_py import PyHex, PySearch
from net import wrap_for_rust
import numpy as np
from tqdm import trange

# Pits player1 against player2. Each player is a function from board state to a valid action
# Returns 1 if player1 wins, else 0
def pit(game, player1, player2, verbose=False):
    player, next_player = player1, player2
    turn = 0
    while game.terminal_value() is None:
        if verbose:
            print(game.to_string())
        player = player1 if turn == 0 else player2
        action = player(game)
        game = game.next_state(action)
        turn = 1 - turn
    result = 1 if turn == 1 else 0
    if verbose:
        print(game.to_string())
        print('Game over. Player {} wins'.format(2 - result))
    return result

# random_players takes a valid action with equal probability
def random_player(game):
    valids = game.valid_actions()
    return random.choice(valids)

def human_player(game):
    valids = game.valid_actions()
    action = None
    while action not in valids:
        if action is not None:
            print('Valid actions are: ', valids)
        try:
            action = int(input('Please input a valid action:'))
        except ValueError:
            action = -1
    return action

def create_mcts_player(net, mcts_iterations=1000):
    search = PySearch()
    net = wrap_for_rust(net)
    def player(game):
        return search.get_action(game, mcts_iterations, net)
    return player

def create_shallow_player(net, verbose=False):
    net = wrap_for_rust(net)
    def player(game):
        valids = game.valid_actions()
        _, policy = net(game.to_array())
        for i in range(len(policy)):
            if i not in valids:
                policy[i] = 0
        options = sorted(enumerate(policy), key=lambda e: -e[1])
        if verbose:
            print(options[:4])
        return options[0][0]
    return player

# Plays n_games, with first move split between half the games
# Returns the win rate for player1 against player 2
def evaluate_network(game, net, n_games, mcts_iterations=0):
    def create_player():
        return create_mcts_player(None, 1000)
        if mcts_iterations > 0:
            return create_mcts_player(net, mcts_iterations)
        else:
            return create_shallow_player(net)
    baseline = create_mcts_player(None, 1000)
    n_games_half = n_games // 2
    wins_first_move = 0
    wins_second_move = 0
    for _ in trange(n_games_half):
        wins_first_move += pit(game.copy(), create_player(), baseline, verbose=False)
        wins_second_move += 1 - pit(game.copy(), baseline, create_player(), verbose=False)
    #return wins / (n_games_half * 2)
    return (wins_first_move / n_games_half), (wins_second_move / n_games_half)

def run_evaluate(args):
    import torch
    config, model_number, mcts_iterations, n_games = args

    model_path = '../runs/{}/models/{}.pt'.format(config['directory'], model_number)
    net = torch.load(model_path)
    game = PyHex(config['board_size'])
    def create_player():
        if mcts_iterations > 0:
            return create_mcts_player(net, mcts_iterations)
        else:
            return create_shallow_player(net)
    def create_baseline():
        if mcts_iterations > 0:
            return create_mcts_player(None, mcts_iterations)
        else:
            return random_player
    
    n_games_half = n_games // 2
    wins_first_move = 0
    wins_second_move = 0
    for _ in range(n_games_half):
        wins_first_move += pit(game.copy(), create_player(), create_baseline(), verbose=False)
        wins_second_move += 1 - pit(game.copy(), create_baseline(), create_player(), verbose=False)
    winrate = (wins_second_move + wins_second_move) / (n_games_half * 2)
    return model_number, winrate

if __name__ == '__main__':
    import pathlib
    import os
    import argparse
    import json
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    parser = argparse.ArgumentParser(description='Evaluate a hex agent')
    parser.add_argument('--processes', type=int, default=cpu_count() // 2,
                        help='number of processes to spin up')
    parser.add_argument('--games', type=int, default=100,
                        help='number of evaluation games per model')
    parser.add_argument('--mcts-iterations', type=int, default=0,
                        help='mcts iterations per move, or 0 for shallow network against random')
    parser.add_argument('--dir', nargs='+', required=True,
                        help='models to evaluate')
    args = parser.parse_args()

    for dir in args.dir:
        print('Loading {} for evaluation...'.format(dir))
    
        base_path = '../runs/{}'.format(dir)
        models_path = '{}/models'.format(base_path)
        config_path = '{}/config.json'.format(base_path)
        out_path = '{}/eval-{}.txt'.format(base_path, args.mcts_iterations)

        if not pathlib.Path(config_path).is_file():
            print('  Config file does not exist! Skipping')
            continue

        with open(config_path) as config_file:
            config = json.load(config_file)
        
        model_numbers = [int(os.path.splitext(f)[0]) for f in os.listdir(models_path)]

        with Pool(args.processes) as pool:
            eval_args = [(config, model_number, args.mcts_iterations, args.games) for model_number in model_numbers]
            eval_results = []
            for result in tqdm(pool.imap(run_evaluate, eval_args), total=len(eval_args), unit='models', desc='  Testing models'):
                eval_results.append(result)
        
        with open(out_path, 'w') as out_file:
            for model_number, winrate in sorted(eval_results):
                print('{}, {}'.format(model_number, winrate), file=out_file)
        print('  Wrote to {}'.format(out_path))        
