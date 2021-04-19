config = {
    'directory': '7x7_final',
    'board_size': 7,
    'transfer_from': None,  # Set this to a different run directory if you wanna do transfer learning. Make sure that the conv layers match!
    'net_type': 'resnet_with_padding',
    'net_blocks': [[16, 16, 16, 16], [16, 16], [16, 16]],
    'self_play_iterations': 50,
    'games_per_iteration': 50,
    'mcts_iterations': 2000,

    'temperature': 1.25, # During self play, action probabilities are raised to the (1 / temperature) before sampling
    'train_epochs': 20,
    'train_batch_size': 64,
    'eval_games': 20,
}
