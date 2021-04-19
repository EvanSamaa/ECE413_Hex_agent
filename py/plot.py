from matplotlib import pyplot as plt
import numpy as np

path = '../runs/7x7_transfer_final/eval-0.txt'
data = np.loadtxt(path, delimiter=',')

plt.plot(data[:, 0], data[:, 1])
plt.ylim([0,1])
plt.savefig('test.png')
