# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from .auto_grad import Tensor
import src.auto_grad


def init_HE(in_features, out_features, dtype='float32'):
    '''
    to initialize the weights of the network
    '''
    std = math.sqrt(2.0 / in_features)
    weight = torch.empty(out_features, in_features, dtype=dtype) # create a weight tensor without initialization
    torch.nn.init.normal_(weight, mean=0, std=std)
    return weight


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value):
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value):
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self):
        return _child_modules(self.__dict__)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def forward(self,*args, **kwargs):
        pass


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
    def forward(self, X):
        size = reduce(lambda a, b: a * b, X.shape)
        return X.reshape((X.shape[0], size // X.shape[0]))


class ReLU(nn.Module):
    def forward(self, x):
        return src.auto_grad.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
            return x