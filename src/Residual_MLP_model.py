# -*- coding: utf-8 -*-
from src.MyModule import Module, Linear, ReLU
import torch
import torch.nn as nn
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x += residual  # 残差连接
        x = self.relu(x)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure input tensor x is converted to float type if necessary
        if x.dtype != torch.float32:
            x = x.float()
        x = self.fc1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.fc2(x)
        return x