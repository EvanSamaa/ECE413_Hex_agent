import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mcts_py import PyHex, PySearch
from net import Net, wrap_for_rust
from players import pit, create_mcts_player, human_player

BOARD_SIZE = 6
MCTS_ITERATIONS = BOARD_SIZE ** 4 # Iterations used for self-play and evaluation
N_EVAL_GAMES = 40 # Number of games played for each evaluation

N_SELF_PLAY_ITERATIONS = 100
N_GAMES = 100

def collect_training_examples(n_games=100, board_size=5, mcts_iterations=1000, value_fn=None):
    results = []
    for i in range(n_games):
        search_tree = PySearch()
        states, values, policy = search_tree.self_play(
            board_size,
            mcts_iterations,
            value_fn,
        )
        results.append((states, values, policy))
    states = torch.tensor(np.concatenate([s for s, v, p in results]))
    values = torch.tensor(np.concatenate([v for s, v, p in results]))
    policy = torch.tensor(np.concatenate([p for s, v, p in results]))
    # TODO: shuffle, extract a validation set
    return augment_examples(states, values, policy)

def augment_examples(states, values, policy):
    n = states.shape[2]
    augmented = []
    for flip_player in [False, True]:
        for flip_pieces in [False, True]:
            s, v, p = states, values, policy
            if flip_player:
                s, v = torch.flip(torch.transpose(s, 2, 3), [1]), -v
                p = torch.transpose(p.reshape((-1, n, n)), 1, 2).reshape((-1, n ** 2))
            if flip_pieces:
                s = torch.flip(s, [2, 3])
                p = torch.flip(p, [1])
            augmented.append((s, v, p))
    states = torch.cat([s for s, v, p in augmented])
    values = torch.cat([v for s, v, p in augmented])
    policy = torch.cat([p for s, v, p in augmented])
    return states, values, policy

def train(net, states, values, policy):
    value_loss_fn = nn.MSELoss()
    opt = optim.Adam(net.parameters())
    
    losses = []

    bs = 32
    epochs = 20
    for i in range(math.floor(epochs * states.shape[0] / bs)):
        samp = np.random.randint(0, states.shape[0], size=(bs,))
        X = states[samp]
        target_values = values[samp].reshape((-1, 1))
        target_policy = policy[samp]

        opt.zero_grad()   # zero the gradient buffers
        output_values, output_policy = net(X)

        value_loss = value_loss_fn(output_values, target_values)
        policy_loss = -torch.sum(target_policy * torch.log(output_policy)) / bs
        loss = value_loss + policy_loss
        loss.backward()
        opt.step()    # Does the update

        losses.append((value_loss.item(), policy_loss.item()))
    
    print('Loss V: {}, Loss P: {}'.format(*[sum(l) / len(l) for l in zip(*losses)]))

# Pits the network against MCTS with random rollouts for (2 * n_games)
def evaluate(net, n_games):
    n_games_half = n_games // 2
    wins = 0
    for _ in range(n_games_half):
        player1 = create_mcts_player(net, mcts_iterations=MCTS_ITERATIONS)
        player2 = create_mcts_player(None, mcts_iterations=MCTS_ITERATIONS)
        wins += pit(PyHex(BOARD_SIZE), player1, player2)
    for _ in range(n_games_half):
        player2 = create_mcts_player(net, mcts_iterations=MCTS_ITERATIONS)
        player1 = create_mcts_player(None, mcts_iterations=MCTS_ITERATIONS)
        wins += 1 - pit(PyHex(BOARD_SIZE), player1, player2)
    return wins / (n_games_half * 2)

net = Net(BOARD_SIZE)

print('Evaluating...')
print('Win rate: ', evaluate(net, N_EVAL_GAMES))

for i in range(N_SELF_PLAY_ITERATIONS):
    print('Starting self play iteration {}/{}'.format(i+1, N_SELF_PLAY_ITERATIONS))

    value_fn = None if i == 0 else net # Use random rollouts for the first iteration
    states, values, policy = collect_training_examples(
        board_size=BOARD_SIZE,
        n_games=N_GAMES,
        mcts_iterations=MCTS_ITERATIONS,
        value_fn=wrap_for_rust(value_fn),
    )

    print('Collected {} samples'.format(states.shape[0]))

    print('Training...')
    train(net, states, values, policy)

    print('Evaluating...')
    print('Win rate: ', evaluate(net, N_EVAL_GAMES))

# while True:
#     player1 = human_player
#     player2 = create_mcts_player(net)
#     pit(PyHex(BOARD_SIZE), player1, player2, verbose=True)
