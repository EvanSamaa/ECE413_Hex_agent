config = {
    'directory': '5x5source',
    'board_size': 5,
    'transfer_from': None,  # Set this to a different run directory if you wanna do transfer learning. Make sure that the conv layers match!
    'net_type': 'resnet_with_padding',
    'net_blocks': [[16, 16, 16, 16]],
    'self_play_iterations': 4,
    'games_per_iteration': 20,
    'mcts_iterations': 600,
    'temperature': 1.25, # During self play, action probabilities are raised to the (1 / temperature) before sampling
    'train_epochs': 50,
    'train_batch_size': 64,
    'eval_games': 8,
}
