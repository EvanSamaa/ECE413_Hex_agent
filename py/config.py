config = {
    'directory': 'test-6x6-conv',

    'board_size': 6,
    'net_type': 'conv',
    'net_conv_channels': [16, 16, 16, 16],

    'self_play_iterations': 80,
    'games_per_iteration': 100,
    'mcts_iterations': 6**4,

    'train_epochs': 20,
    'train_batch_size': 32,

    'eval_games': 10,
}
