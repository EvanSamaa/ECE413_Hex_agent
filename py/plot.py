from matplotlib import pyplot as plt
import numpy as np

# Plots of win rate versus training iterations
training_plots = [
    {
        'title': '5x5',
        'models': ['5x5_final'],
        'eval-iterations': [0, 100, 500],
    },
    {
        'title': '7x7',
        'models': ['7x7_final', '7x7_transfer_final'],
        'eval-iterations': [0, 100],
    },
    {
        'title': '8x8',
        'models': ['8x8_Raw_final', '8x8_transfer_final'],
        'eval-iterations': [0, 100],
    },
]

for plot_config in training_plots:
    plt.figure()
    for its in plot_config['eval-iterations']:
        for model in plot_config['models']:
            path = '../runs/{}/eval-{}.txt'.format(model, its)
            data = np.loadtxt(path, delimiter=',')
            plt.plot(data[:, 0], data[:, 1], label='{} vs {}'.format(model, 'random' if its == 0 else 'mcts-{}'.format(its)))
    plt.ylim([0,1])
    plt.title(plot_config['title'])
    plt.xlabel('Self play iterations')
    plt.ylabel('Win rate against baseline')
    plt.legend(loc="center right")
    plt.savefig('../plots/{}.png'.format(plot_config['title']))
