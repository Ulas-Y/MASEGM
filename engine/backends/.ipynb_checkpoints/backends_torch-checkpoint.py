# engine/backends/backend_torch.py

import torch
from .backend import Backend


class TorchBackend(Backend):
    def __init__(self, device=None, dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float64
        self.device = device
        self.dtype = dtype

    # basic ops
    def asarray(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def log(self, x):
        return torch.log(x)

    def exp(self, x):
        return torch.exp(x)

    def maximum(self, x, val):
        return torch.clamp(x, min=val)

    def clip(self, x, min_val, max_val):
        return torch.clamp(x, min=min_val, max=max_val)

    # B-arithmetic
    def b_add(self, x, y):
        return x * y

    def b_sub(self, x, y):
        return x / y

    def b_mult(self, x, y):
        return x ** y

    def b_div(self, x, y, eps=1e-30):
        eps_t = torch.tensor(eps, device=self.device, dtype=self.dtype)
        x = torch.clamp(x, min=eps)
        one = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y = torch.where((y <= 0) | torch.isclose(y, one), eps_t, y)
        return torch.log(x) / torch.log(y)

    # grid ops
    def log_gradient(self, field, eps: float = 1e-12):
        f = torch.clamp(field, min=eps)
        logf = torch.log(f)
        gy = 0.5 * (torch.roll(logf, shifts=-1, dims=0) - torch.roll(logf, shifts=1, dims=0))
        gx = 0.5 * (torch.roll(logf, shifts=-1, dims=1) - torch.roll(logf, shifts=1, dims=1))
        return gy, gx

    def log_laplacian(self, field, eps: float = 1e-12):
        f = torch.clamp(field, min=eps)
        logf = torch.log(f)
        lap = (
            -4.0 * logf
            + torch.roll(logf, 1, 0)
            + torch.roll(logf, -1, 0)
            + torch.roll(logf, 1, 1)
            + torch.roll(logf, -1, 1)
        )
        return lap

    def divergence(self, Fy, Fx):
        dFy_dy = 0.5 * (torch.roll(Fy, -1, 0) - torch.roll(Fy, 1, 0))
        dFx_dx = 0.5 * (torch.roll(Fx, -1, 1) - torch.roll(Fx, 1, 1))
        return dFy_dy + dFx_dx
