# config = {
#     'directory': '6x6-res-4x2',
#     'board_size': 6,
#     'net_type': 'resnet_with_padding',
#     'net_blocks': [[16, 16], [16, 16], [16, 16], [16, 16]],
#     'self_play_iterations': 100,
#     'games_per_iteration': 100,
#     'mcts_iterations': 1000,
#     'temperature': 1.3, # During self play, action probabilities are raised to the (1 / temperature) before sampling
#     'train_epochs': 10,
#     'train_batch_size': 32,
#     'eval_games': 10,
# }

config = {
    'directory': '9x9-res-4x2',
    'board_size': 9,
    'net_type': 'resnet_with_padding',
    'net_blocks': [[16, 16], [16, 16], [16, 16], [16, 16]],
    'self_play_iterations': 20,
    'games_per_iteration': 20,
    'mcts_iterations': 750,
    'temperature': 1.25,
    'train_epochs': 50,
    'train_batch_size': 64,
    'eval_games': 25,
}
Transfer_source_config = {
    'directory': '../runs/6x6-res-4x2/models/80.pt',
    'board_size': 9,
    'net_type': 'resnet_with_padding',
    'net_blocks': [[16, 16], [16, 16], [16, 16], [16, 16]],
    'self_play_iterations': 20,
    'games_per_iteration': 20,
    'mcts_iterations': 750,
    'temperature': 1.25,
    'train_epochs': 50,
    'train_batch_size': 64,
    'eval_games': 25,
}
