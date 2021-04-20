from matplotlib import pyplot as plt
import numpy as np

image_format = 'png'

# Plots of win rate versus training iterations
training_plots = [
    {
        'title': '5x5',
        'models': ['5x5_final'],
        'labels': ['5x5 agent'],
        'formats': ['-'],
        'eval-iterations': [0, 100, 500],
        'colors': ['black', 'gray', 'lightgray'],
    },
    {
        'title': '7x7',
        'models': ['7x7_final', '7x7_transfer_final'],
        'labels': ['7x7 scratch agent', '7x7 transfer agent'],
        'formats': ['-', '--'],
        'eval-iterations': [0, 100],
        'colors': ['black', 'gray'],
    },
    {
        'title': '8x8',
        'models': ['8x8_Raw_final', '8x8_transfer_final'],
        'labels': ['8x8 scratch agent', '8x8 transfer agent'],
        'formats': ['-', '--'],
        'eval-iterations': [0, 100],
        'colors': ['black', 'gray'],
    },
]

for plot_config in training_plots:
    plt.figure()
    for its, color in zip(plot_config['eval-iterations'], plot_config['colors']):
        for model, model_label, format in zip(plot_config['models'], plot_config['labels'], plot_config['formats']):
            path = '../runs/{}/eval-{}.txt'.format(model, its)
            data = np.loadtxt(path, delimiter=',')
            label = '{} vs {}'.format(model_label, 'random' if its == 0 else 'MCTS ({} iterations)'.format(its))
            plt.plot(data[:, 0], data[:, 1], format, color=color, label=label)
    plt.ylim([0, 1])
    plt.xlim([0, 50])
    plt.xlabel('Self play iterations')
    plt.ylabel('Win rate against baseline')
    plt.legend(loc="center right")
    plt.savefig('../plots/{}.{}'.format(plot_config['title'], image_format, bbox_inches='tight'))
