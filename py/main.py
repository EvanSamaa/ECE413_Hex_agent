from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import lru_cache
import numpy as np
import torch

from net import Net, Differential_padding_Net, ResNet, ResNet_with_Padding
from self_play import self_play
from train import train
from evaluate import pit, create_mcts_player
from mcts_py import PyHex
from config import config

model_dict = {
    'conv': Net,
    'resnet': ResNet,
    'resnet_with_padding': ResNet_with_Padding,
    'differential_padding': Differential_padding_Net,
}

class Training_Recorder:
    def __init__(self, save_dir, data_cols=2, epoch=100000):
        self.save_dir = save_dir
        self.data = np.zeros((epoch, data_cols))
        self.data_cols = data_cols
    def log(self, epoch, vals):
        for i in range(self.data_cols):
            self.data[epoch, i] = vals[i]
    def save(self):
        self.data
        np.save(self.save_dir, self.data)

@lru_cache(maxsize=None)
def load_net_eval(path):
    return torch.load(path)

def load_net_train(path):
    if path:
        return torch.load(path)
    else:
        return model_dict[config["net_type"]](config)

def run_self_play(net_path):
    net = load_net_eval(net_path) if net_path else None
    return self_play(config['board_size'], config['mcts_iterations'], config['temperature'], net)

def run_train(model_path, model_out_path, training_examples):
    net = load_net_train(model_path)
    tensors = [torch.tensor(array) for array in training_examples]
    losses = train(net, tensors, bs=config['train_batch_size'], epochs=config['train_epochs'])
    torch.save(net, model_out_path)
    return losses

def run_evaluate(model_path):
    net = load_net_eval(model_path)
    n_games_half = config['eval_games'] // 2
    wins = 0
    for _ in range(n_games_half):
        player1 = create_mcts_player(net, mcts_iterations=config['mcts_iterations'])
        player2 = create_mcts_player(None, mcts_iterations=config['mcts_iterations'])
        wins += pit(PyHex(config['board_size']), player1, player2)
    for _ in range(n_games_half):
        player2 = create_mcts_player(net, mcts_iterations=config['mcts_iterations'])
        player1 = create_mcts_player(None, mcts_iterations=config['mcts_iterations'])
        wins += 1 - pit(PyHex(config['board_size']), player1, player2)
    return wins / (n_games_half * 2)

if __name__ == '__main__':
    import pathlib
    import os
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train a hex agent')
    parser.add_argument('--processes', type=int, default=3,
        help='number of processes to spin up')
    parser.add_argument('--force', default=False, action='store_true',
        help='force training even if the config has changed')
    args = parser.parse_args()

    base_path = '../runs/{}'.format(config['directory'])
    models_path = '{}/models'.format(base_path)
    config_path = '{}/config.json'.format(base_path)
    train_result_path = '{}/train_result.npy'.format(base_path)

    # Save the config, and check if a different one exists
    if pathlib.Path(config_path).is_file():
        with open(config_path, 'r') as config_file:
            existing_config = json.load(config_file)
            if existing_config != config:
                print('Found an existing config with different values:')
                for k, v in config.items():
                    if existing_config[k] != v:
                        print('  {}: {} -> {}'.format(k, existing_config[k], v))
                if args.force:
                    print('Overwriting with new config...')
                else:
                    print('To overwrite and train anyway, run again with --force')
                    exit()
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4, separators=(',', ': '))

    print('Config:')
    for k, v in config.items():
        print('  {}: {}'.format(k, v))
    storage = Training_Recorder(train_result_path, 3, config["train_epochs"])
    print('Number of processes:', args.processes)
    pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
    last_model = max(int(os.path.splitext(f)[0]) for f in os.listdir(models_path)) if os.listdir(models_path) else 0

    if last_model > 0:
        print('Picking up after {} iterations!'.format(last_model))

    training_examples = None

    for i in range(last_model, config['self_play_iterations']):
        print('Self play iteration {}/{}'.format(i + 1, config['self_play_iterations']))
        model_path = '{}/{}.pt'.format(models_path, i - 1) if i - 1 > 0 else None
        self_play_results = []
        with Pool(args.processes) as pool:
            # TRAIN
            training_result = None
            if training_examples is None:
                print('  No training examples yet, skipping training')
            else:
                print('  Training in parallel...')
                model_out_path = '{}/{}.pt'.format(models_path, i)
                training_result = pool.apply_async(run_train, (model_path, model_out_path, training_examples))

            # EVALUATE
            evaluate_result = None
            if model_path:
                print('  Evaluating in parallel...')
                evaluate_result = pool.apply_async(run_evaluate, (model_path,))

            # SELF PLAY
            if model_path is None:
                print('  No trained model this iteration, using random rollouts')
            self_play_args = [model_path] * config['games_per_iteration']
            for result in tqdm(pool.imap(run_self_play, self_play_args), total=config['games_per_iteration'], unit='games', desc='  Getting self play examples'):
                self_play_results.append(result)

            # Join results
            if training_result:
                value_loss, policy_loss = training_result.get()
                print('  Training loss V: {:.6f}, P: {:.6f}'.format(
                    sum(value_loss) / len(value_loss),
                    sum(policy_loss) / len(policy_loss),
                    ))
            if evaluate_result:
                win_rate = evaluate_result.get()
                print('  Win rate:', win_rate)

            # Log
            if training_result and evaluate_result:
                storage.log(i, [win_rate, sum(value_loss) / len(value_loss), sum(policy_loss) / len(policy_loss)])
                storage.save()

        states, values, policy = [np.concatenate(arrays) for arrays in zip(*self_play_results)]
        training_examples = states, values, policy
