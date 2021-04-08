config = {
    'directory': '8x8-conv2',

    'board_size': 8,
    'net_type': 'conv',
    'net_conv_channels': [16, 16, 16, 16],

    'self_play_iterations': 50,
    'games_per_iteration': 100,
    'mcts_iterations': 800,
    'temperature': 1.3, # During self play, action probabilities are raised to the (1 / temperature) before sampling

    'train_epochs': 10,
    'train_batch_size': 32,

    'eval_games': 10,
}
