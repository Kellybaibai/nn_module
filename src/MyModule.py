# -*- coding: utf-8 -*-
# @Time: 2024/6/22 16:06
# @Author: Kellybai
# @File: MyModule.py
# Have a nice day!

import math
import torch
import torch.nn as nn
import numpy as np


def init_HE(in_features, out_features, dtype='float32'):
    '''
    to initialize the weights of the network
    '''
    std = math.sqrt(2.0 / in_features)
    weight = torch.empty(out_features, in_features, dtype=dtype) # create a weight tensor without initialization
    torch.nn.init.normal_(weight, mean=0, std=std)
    return weight


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype='float32'):
        super(Linear, self).__init__() # initialize the  father class nn.Module too
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init_HE(self.in_features, self.out_features)
        if bias:
            self.bias = torch.Tensor(np.zeros(self.out_features))
        else:
            self.bias = None
    def forward(self, X):
        X_out = X @ self.weight  # the multiplication of matrix
        if self.bias:
            return X_out + self.bias.broadcast_to(X_out.shape)
        return X_out



class Flatten(nn.Module):
    def __init__(self):
        pass

class ReLU(nn.Module):
    def __init__(self):
        pass

class Sigmoid(nn.Module):
    def __init__(self):
        pass

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        pass

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        pass

class MSELoss(nn.Module):
    def __init__(self):
        pass

class BatchNorm1d(nn.Module):
    def init(self):
        pass

class LayerNorm1d(nn.Module):
    def __init__(self):
        pass

class Dropout(nn.Module):
    def __init__(self):
        pass

class Sequential(nn.Module):
    def __init__(self, *modules):
        super.__init__()
        self.modules = modules
    def forward(self, X):
        for module in self.modules:
            X = module(X)
        return X

class Residual(nn.Module):
    def __init__(self):
        pass