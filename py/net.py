import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()

        conv_channels = [2, 16, 16]

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=3, padding=1)
            for i in range(len(conv_channels) - 1)
        ])

        self.value_head = nn.ModuleList([
            nn.Linear(n * n * conv_channels[-1], n * n),
            nn.Linear(n * n, 1),
        ])

        self.policy_head = nn.ModuleList([
            nn.Linear(n * n * conv_channels[-1], n * n),
        ])
    
    def forward(self, x):
        # Allows passing numpy arrays
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Allows passing single states or a batch
        if len(x.shape) == 3:
            x = x.reshape((1, *x.shape))

        # Convolutions
        bs = x.shape[0]
        for layer in self.convs:
            x = F.relu(layer(x))
        x = x.reshape((bs, -1))

        # Value head
        vx = x
        for i, layer in enumerate(self.value_head):
            vx = layer(vx)
            if i < len(self.value_head) - 1:
                vx = F.relu(vx)
        vx = torch.sigmoid(vx) * 2 - 1

        # Policy head
        px = x
        for i, layer in enumerate(self.policy_head):
            px = layer(px)
            if i < len(self.policy_head) - 1:
                px = F.relu(px)
        px = px.softmax(dim=1)

        return vx, px

def wrap_for_rust(net):
    if net is None:
        return None
    def model(state):
        value, policy = net(state)
        return value.item(), policy.squeeze().tolist()
    return model
