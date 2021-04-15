config = {
    'directory': '5x5-res-4x2',
    'board_size': 5,
    'net_type': 'resnet_with_padding',
    'net_blocks': [[16, 16], [16, 16], [16, 16], [16, 16]],
    'self_play_iterations': 100,
    'games_per_iteration': 100,
    'mcts_iterations': 1000,
    'temperature': 1.3, # During self play, action probabilities are raised to the (1 / temperature) before sampling
    'train_epochs': 10,
    'train_batch_size': 32,
    'eval_games': 10,
}