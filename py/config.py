config = {
    'directory': '8x8-res',
    'board_size': 8,
    'net_type': 'resnet',
    'net_blocks': [[32, 32, 32], [32, 32, 32], [32, 32, 32]],
    'self_play_iterations': 100,
    'games_per_iteration': 50,
    'mcts_iterations': 600,
    'temperature': 1.3, # During self play, action probabilities are raised to the (1 / temperature) before sampling
    'train_epochs': 5,
    'train_batch_size': 32,
    'eval_games': 10,
}
