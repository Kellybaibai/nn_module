# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy
from collections import defaultdict
from .auto_grad import Tensor

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for i, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            param.data = param.data - grad * self.lr


class SGD_WeightDecay(Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.params):
            if self.weight_decay > 0:
                grad = param.grad.data + self.weight_decay * param.data
            else:
                grad = param.grad.data
            param.data = param.data - grad * self.lr


class Momentum(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)

    def step(self):
        for i, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            param.data = param.data - self.lr * self.u[param]



class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        self.t += 1
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)
            unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)