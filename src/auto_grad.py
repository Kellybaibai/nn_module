# -*- coding: utf-8 -*-

import numpy as np

TENSOR_COUNTER = 0


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    for in_node in node.inputs:
        topo_sort_dfs(in_node, visited, topo_order)
    topo_order.append(node)
    visited.add(node)

def find_topo_sort(node_list):
    """
    Given a list of nodes, return a topological sort list of nodes ending in them.
    """
    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def compute_gradient_of_variables(output_tensor, out_grad):
    """
    the Reverse AD algorithm
    """
    # save for partial adjoint
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        # sum up partial ajoints
        node.grad = sum(node_to_output_grads_list[node])
        if node.is_leaf():
            # Leaf node
            continue
        # compute partial ajoints for input node
        for i, grad in enumerate(node.op.gradient(node.grad, node)):
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad)


class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args):
        raise NotImplementedError()

    def gradient(self, out_grad, node):
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad, node):
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class Value:
    def _init(self, op, inputs, *, num_outputs=1, cached_data=None, requires_grad=None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
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

    def __del__(self):
        print('Deleting object of Value')
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op, inputs):
        value = cls.__new__(cls)
        value._init(op, inputs)
        value.realize_cached_data()
        return value


class Tensor(Value):
    def __init__(self, array, *, dtype=None, requires_grad=True, **kwargs):
        cached_data = array.realize_cached_data()
        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def from_numpy(numpy_array, dtype):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op, inputs):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor_data = data
        else:
            tensor_data = data.realize_cached_data()
        tensor._init(None, [], cached_data=tensor_data, requires_grad=requires_grad)
        return tensor


    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else Tensor(np.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)


class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

'''
compute: computes the forward pass
gradient: computes the backward pass 
'''

class EWiseAdd(TensorOp):
    def compute(self, a, b):
        return a + b

    def gradient(self, out_grad, node):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a + self.scalar

    def gradient(self, out_grad, node):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a, b):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a):
        return np.power(a, self.scalar)

    def gradient(self, out_grad, node):
        return node.inputs[0]**(self.scalar - 1) * out_grad * self.scalar


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return np.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, - lhs * out_grad / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return np.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return np.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return np.swapaxes(a, a.ndim - 2, a.ndim - 1)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return np.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return np.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        return np.sum(a, self.axes)

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return np.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return np.negative(a)

    def gradient(self, out_grad, node):
        return - out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return np.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return np.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        out = np.copy(a)
        out[a < 0] = 0
        return out

    def gradient(self, out_grad, node):
        out = node.realize_cached_data().copy()
        out[out > 0] = 1
        return out_grad * Tensor(out)


def relu(a):
    return ReLU()(a)