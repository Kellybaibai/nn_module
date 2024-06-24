# -*- coding: utf-8 -*-
# @Time: 2024/6/24 17:37
# @Author: Kellybai
# @File: cp_graph.py
# Have a nice day!

class Op:
    def compute(self,*args):
        pass
    def gradient(self, out_grad, node):
        pass

class Value:
    def __init__(self, op, inputs, *, num_outputs=1, cached_data=None, requires_grad=None):
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def is_leaf(self):
        return self.op is None

    def realize_cached_data(self):
        if self.is_leaf() or self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data