# engine/backends/backend_numpy.py

import numpy as np
from .backend import Backend


class NumpyBackend(Backend):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    # basic ops
    def asarray(self, x):
        return np.asarray(x, dtype=self.dtype)

    def full(self, shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=self.dtype if dtype is None else dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=self.dtype if dtype is None else dtype)

    def log(self, x):
        return np.log(x)

    def exp(self, x):
        return np.exp(x)

    def maximum(self, x, val):
        return np.maximum(x, val)

    def clip(self, x, min_val, max_val):
        return np.clip(x, min_val, max_val)

    # B-arithmetic
    def b_add(self, x, y):
        return x * y

    def b_sub(self, x, y):
        return x / y

    def b_mul(self, x, y):
        return x ** y

    def b_div(self, x, y, eps=1e-30):
        x = np.asarray(x, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype)
        x = np.maximum(x, eps)
        y = np.where((y <= 0) | (np.isclose(y, 1.0)), eps, y)
        return np.log(x) / np.log(y)

    # grid ops = directly based on your existing code
    def log_gradient(self, field, eps: float = 1e-12):
        f = np.maximum(field, eps)
        logf = np.log(f)
        gy = 0.5 * (np.roll(logf, -1, axis=0) - np.roll(logf, 1, axis=0))
        gx = 0.5 * (np.roll(logf, -1, axis=1) - np.roll(logf, 1, axis=1))
        return gy, gx

    def log_laplacian(self, field, eps: float = 1e-12):
        f = np.maximum(field, eps)
        logf = np.log(f)
        lap = (
            -4.0 * logf
            + np.roll(logf, 1, axis=0)
            + np.roll(logf, -1, axis=0)
            + np.roll(logf, 1, axis=1)
            + np.roll(logf, -1, axis=1)
        )
        return lap

    def divergence(self, Fy, Fx):
        dFy_dy = 0.5 * (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0))
        dFx_dx = 0.5 * (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1))
        return dFy_dy + dFx_dx

    def laplacian(self, field):
        return (
            -4.0 * field
            + np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        )
