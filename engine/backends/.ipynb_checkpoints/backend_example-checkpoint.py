# backend.py

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract base class for numerical backend (NumPy or Torch)."""

    @abstractmethod
    def asarray(self, x):
        ...

    @abstractmethod
    def log(self, x):
        ...

    @abstractmethod
    def exp(self, x):
        ...

    @abstractmethod
    def maximum(self, x, val):
        ...

    @abstractmethod
    def clip(self, x, min_val, max_val):
        ...

    # ---- B-arithmetic ----
    @abstractmethod
    def b_add(self, x, y):
        ...

    @abstractmethod
    def b_sub(self, x, y):
        ...

    @abstractmethod
    def b_mul(self, x, y):
        ...

    @abstractmethod
    def b_div(self, x, y):
        ...

    # ---- Grid ops ----
    @abstractmethod
    def log_gradient(self, field, eps: float = 1e-12):
        ...

    @abstractmethod
    def log_laplacian(self, field, eps: float = 1e-12):
        ...

    @abstractmethod
    def divergence(self, Fy, Fx):
        ...
