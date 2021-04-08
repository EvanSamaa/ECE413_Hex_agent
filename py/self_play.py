
import numpy as np
from net import wrap_for_rust
from mcts_py import PySearch

# Gets training examples from a single game of self play
# If net is None, the value estimate is random rollouts
def self_play(board_size, mcts_iterations, temperature, net=None):
    search_tree = PySearch()
    states, values, policy = search_tree.self_play(
        board_size,
        mcts_iterations,
        temperature,
        wrap_for_rust(net),
    )
    return augment_examples(states, values, policy)

# Augments training examples through board symmetries
#   Switching players negates the value
#   Rotating the board 180degrees does not change the value
def augment_examples(states, values, policy):
    n = states.shape[2]
    augmented = []
    for flip_player in [False, True]:
        for flip_pieces in [False, True]:
            s, v, p = states, values, policy
            if flip_player:
                s, v = np.flip(np.swapaxes(s, 2, 3), [1]), -v
                p = np.swapaxes(p.reshape((-1, n, n)), 1, 2).reshape((-1, n ** 2))
            if flip_pieces:
                s = np.flip(s, [2, 3])
                p = np.flip(p, [1])
            augmented.append((s, v, p))
    states, values, policy = [np.concatenate(arrays) for arrays in zip(*augmented)]
    return states, values, policy
