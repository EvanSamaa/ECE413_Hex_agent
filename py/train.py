import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Trains the net on the examples, and returns the training value and policy loss
def train(net, examples, bs=32, epochs=20):
    value_loss_fn = nn.MSELoss()
    opt = optim.Adam(net.parameters())

    losses = []

    states, values, policy = examples
    all_samps = np.random.choice(epochs * states.shape[0], (epochs * states.shape[0],), replace=False)
    for i in range(math.floor(epochs * states.shape[0] / bs)):
        # samp = np.random.randint(0, states.shape[0], size=(bs,))
        # X = states[samp]
        # target_values = values[samp].reshape((-1, 1))
        # target_policy = policy[samp]
        samp = all_samps[i * bs: (i + 1) * bs]
        X = states[np.mod(samp, epochs)]
        target_values = values[np.mod(samp, epochs)].reshape((-1, 1))
        target_policy = policy[np.mod(samp, epochs)]


        opt.zero_grad()  # zero the gradient buffers
        output_values, output_policy = net(X)

        value_loss = value_loss_fn(output_values, target_values)
        policy_loss = -torch.sum(target_policy * torch.log(output_policy)) / bs
        loss = value_loss + policy_loss
        loss.backward()
        opt.step()    # Does the update

        losses.append((value_loss.item(), policy_loss.item()))
    return list(zip(*losses))
