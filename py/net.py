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

        self.linears = nn.ModuleList([
            nn.Linear(n * n * conv_channels[-1], n * n),
            nn.Linear(n * n, 1),
        ])
    
    def forward(self, x):
        # Allows passing numpy arrays
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Allows passing single states or a batch
        if len(x.shape) == 3:
            x = x.reshape((1, *x.shape))

        bs = x.shape[0]
        for layer in self.convs:
            x = F.relu(layer(x))
        x = x.reshape((bs, -1))
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
        return torch.sigmoid(x) * 2 - 1
