import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from config import config, transfer_source_config
class ResBlock(nn.Module):
    def __init__(self, channels, res=True):
        super(ResBlock, self).__init__()
        self.res = res
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3, padding=1)
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.convs):
            x = layer(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        if self.res:
            return x + residual
        else:
            return x
class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        n = config['board_size']

        block_channels = config['net_blocks']

        blocks = [
            nn.Conv2d(in_channels=2, out_channels=block_channels[0][0], kernel_size=3, padding=1)
        ]
        for i, channels in enumerate(block_channels):
            blocks.append(ResBlock(channels, res=i>0))
        self.blocks = nn.ModuleList(blocks)

        self.value_head = nn.ModuleList([
            nn.Linear(n * n * block_channels[-1][-1], n * n),
            nn.Linear(n * n, 1),
        ])

        self.policy_head = nn.ModuleList([
            nn.Linear(n * n * block_channels[-1][-1], n * n),
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
        for layer in self.blocks:
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
class ResNet_with_Padding(nn.Module):
    def __init__(self, config):
        super(ResNet_with_Padding, self).__init__()

        n = config['board_size']

        block_channels = config['net_blocks']
        blocks = [
            nn.Conv2d(in_channels=2, out_channels=block_channels[0][0], kernel_size=3, padding=0)
        ]
        for i, channels in enumerate(block_channels):
            blocks.append(ResBlock(channels, res=i > 0))
        self.blocks = nn.ModuleList(blocks)

        self.value_head = nn.ModuleList([
            nn.Linear(n * n * block_channels[-1][-1], n * n),
            nn.Linear(n * n, 1),
        ])

        self.policy_head = nn.ModuleList([
            nn.Linear(n * n * block_channels[-1][-1], n * n),
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
        x0 = torch.nn.functional.pad(x[:, 0:1], [1, 1, 0, 0], mode="constant", value=0)
        x0 = torch.nn.functional.pad(x0, [0, 0, 1, 1], mode="constant", value=1)
        x1 = torch.nn.functional.pad(x[:, 1:], [0, 0, 1, 1], mode="constant", value=0)
        x1 = torch.nn.functional.pad(x1, [1, 1, 0, 0], mode="constant", value=1)
        x = torch.cat([x0, x1], dim=1)
        for layer in self.blocks:
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

def transfer(source_path, config, target_model_class, conv_only=True):
    model_source = torch.load(source_path)
    model_target = target_model_class(config)
    # model = source_model_class(source_model_config)
    # model.load_state_dict(torch.load(source_model_config['directory']))
    if conv_only:
        for (nameA, paramA), (nameB, paramB) in zip(model_source.named_parameters(), model_target.named_parameters()):
            if (paramA.shape == paramB.shape):
                paramB.data = paramA.data
    return model_target


def wrap_for_rust(net):
    if net is None:
        return None
    def model(state):
        with torch.no_grad():
            value, policy = net(state.copy())
        return value.item(), policy.squeeze().tolist()
    return model
# if __name__ == "__main__":
    # here are the test codes
    # transfer(transfer_source_config, config, Differential_padding_Net)
