import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        n = config['board_size']

        conv_channels = [2, *config['net_conv_channels']]

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

class Differential_padding_Net(nn.Module):
    def __init__(self, config):
        super(Differential_padding_Net, self).__init__()

        n = config['board_size']

        conv_channels = [2, *config['net_conv_channels']]

        # self.convs = nn.ModuleList([
        #     nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=3, padding=1)
        #     for i in range(len(conv_channels) - 1)
        # ])
        self.convs = []
        for i in range(0, len(conv_channels) - 1):
            if i > 0:
                self.convs.append(nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=3, padding=1))
            else:
                self.convs.append(
                    nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i + 1], kernel_size=3,
                              padding=0))
        self.convs = nn.ModuleList(self.convs)
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
        x0 = torch.nn.functional.pad(x[:,0:1], [1,1,0,0], mode="constant", value=0)
        x0 = torch.nn.functional.pad(x0, [0,0,1,1], mode="constant", value=1)
        x1 = torch.nn.functional.pad(x[:, 1:], [0, 0, 1, 1], mode="constant", value=0)
        x1 = torch.nn.functional.pad(x1, [1, 1, 0, 0], mode="constant", value=1)
        x = torch.cat([x0, x1], dim=1)
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
        with torch.no_grad():
            value, policy = net(state.copy())
        return value.item(), policy.squeeze().tolist()
    return model
