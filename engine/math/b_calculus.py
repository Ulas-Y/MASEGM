from __future__ import annotations

"""
b_calculus.py

Full implementation of the B-Scale arithmetic, B-derivative, and B-integral
based on your formal definitions.

Author: ME aka "U" the author, innovator, engineer and scientist (concept) + ChatGPT (implementation)
"""

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from engine.backends import select_backend
from engine.backends.backends_numpy import NumpyBackend

if find_spec("torch") is not None:
    from engine.backends.backends_torch import TorchBackend
else:  # pragma: no cover - torch is optional
    TorchBackend = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    import torch

NUMPY_BACKEND = NumpyBackend()
TORCH_BACKEND = TorchBackend() if TorchBackend is not None else None

BACKEND_NAME, xp = select_backend()

if BACKEND_NAME == "torch" and TORCH_BACKEND is not None:
    TORCH_BACKEND = xp
elif BACKEND_NAME == "numpy":
    NUMPY_BACKEND = xp


def _backend(backend=None):
    if backend is None:
        return xp
    if isinstance(backend, str):
        name = backend.lower()
        if name == "numpy":
            return NUMPY_BACKEND
        if name == "torch":
            if TORCH_BACKEND is None:
                raise ImportError("Torch backend requested but torch is not installed.")
            return TORCH_BACKEND
        raise ValueError(f"Unknown backend string: {backend}")
    return backend


def set_backend(backend=None):
    """Override the module-level backend.

    ``backend`` can be a backend instance, ``"numpy"``, or ``"torch"``.
    Returns the backend that is set.
    """

    global xp, BACKEND_NAME
    be = _backend(backend)
    xp = be
    BACKEND_NAME = "torch" if (TORCH_BACKEND is not None and be is TORCH_BACKEND) else "numpy"
    return be


def _as_backend_array(x, backend=None):
    be = _backend(backend)
    return be.asarray(x)

# ============================================================
#  B-ARITHMETIC (Fundamental Operations)
# ============================================================

def b_add(x, y, backend=None):
    """B-addition: x ⊕ y = x * y"""
    be = _backend(backend)
    return be.b_add(_as_backend_array(x, be), _as_backend_array(y, be))


def b_sub(x, y, backend=None):
    """B-subtraction: x ⊖ y = x / y"""
    be = _backend(backend)
    return be.b_sub(_as_backend_array(x, be), _as_backend_array(y, be))


def b_mult(x, y, backend=None):
    """B-multiplication: x ⊗ y = x^y"""
    be = _backend(backend)
    return be.b_mul(_as_backend_array(x, be), _as_backend_array(y, be))


def b_div(x, y, eps=1e-12, backend=None):
    be = _backend(backend)
    return be.b_div(_as_backend_array(x, be), _as_backend_array(y, be), eps=eps)



# ============================================================
#  B-DERIVATIVE
# ============================================================

def b_derivative(f, x, dx=1e-8, eps=1e-30, backend=None):
    """
    Computes D_B f(x) = exp( d/d(ln x) ln f(x) ) using the chosen backend.

    Uses finite differences:
        d/d(ln x) g = (g(x+dx) - g(x-dx)) / ( ln(x+dx) - ln(x-dx) )
    """
    be = _backend(backend)
    x_arr = be.asarray(x)

    xp_arr = x_arr + dx
    xm_arr = x_arr - dx
    xm_arr = be.maximum(xm_arr, be.asarray(dx))  # ensure positive

    def _g(z):
        return be.log(be.maximum(f(z), be.asarray(eps)))

    num = _g(xp_arr) - _g(xm_arr)
    den = be.log(xp_arr) - be.log(xm_arr)

    return be.exp(num / den)


# ============================================================
#  B-INTEGRAL (Continuous, Numeric Approximation)
# ============================================================

def b_integral(f, a, b, n=10000, eps=1e-30, backend=None):
    """
    Compute ∫ f(x) d_B x = exp( ∫ ln(f(x)) d(ln x) ) using the active backend.

    Numeric approach:
        I = exp( sum ln(f(x_i)) * d(ln x_i) )
    
    Parameters:
        f : function
        a,b : integration limits (>0)
        n : number of steps
    """
    if a <= 0 or b <= 0:
        raise ValueError("B-integral domain requires a>0, b>0.")

    be = _backend(backend)
    xs = be.asarray(np.linspace(a, b, n))
    lnf = be.log(be.maximum(f(xs), be.asarray(eps)))

    lnxs = be.log(xs)
    dlnx = lnxs[1:] - lnxs[:-1]

    mid = (lnf[1:] + lnf[:-1]) / 2
    total = (mid * dlnx).sum()
    return be.exp(total)


# ============================================================
#  OPTIONAL: TORCH VERSION (GPU-READY)
# ============================================================

def b_derivative_torch(f, x, dx=1e-6, eps=1e-30):
    import torch

    xp = x + dx
    xm = x - dx
    xm = torch.where(xm <= 0, x, xm)

    g = lambda z: torch.log(torch.clamp(f(z), min=eps))

    num = g(xp) - g(xm)
    den = torch.log(xp) - torch.log(xm)

    return torch.exp(num / den)


def b_integral_torch(f, a, b, n=20000, device=None, dtype=None, eps=1e-30):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float64

    xs = torch.linspace(a, b, n, device=device, dtype=dtype)
    lnf = torch.log(torch.clamp(f(xs), min=eps))
    lnxs = torch.log(xs)
    dlnx = lnxs[1:] - lnxs[:-1]

    mid = (lnf[1:] + lnf[:-1]) / 2
    return torch.exp(torch.sum(mid * dlnx))


# ============================================================
#  OPTIONAL: SYMBOLIC VERSION (with SymPy)
# ============================================================

def b_derivative_sympy(f, x):
    """
    Symbolic B-derivative:
        D_B f(x) = exp( d/d(ln x) ln f(x) )
    """
    import sympy as sp
    ln_x = sp.log(x)
    return sp.exp(sp.diff(sp.log(f), ln_x))


# ... your existing B-calculus code ...


def log_gradient(field, eps: float = 1e-12, backend=None):
    """
    Gradient of ln(field) using central differences with periodic-ish boundaries.

    Returns (gy, gx) arrays.
    """
    be = _backend(backend)
    return be.log_gradient(_as_backend_array(field, be), eps=eps)


def b_gradient(field, eps: float = 1e-12, backend=None):
    """
    B-gradient: exp(grad ln(field)).

    Mostly for future use (forces, flows). For diffusion we mainly
    use log_gradient/log_laplacian.
    """
    be = _backend(backend)
    gy, gx = log_gradient(field, eps=eps, backend=be)
    return be.exp(gy), be.exp(gx)


def log_laplacian(field, eps: float = 1e-12, backend=None):
    """
    Discrete Laplacian of ln(field) on a 2D grid with simple 5-point stencil.

    This is the main operator we use for B-diffusion.
    """
    be = _backend(backend)
    return be.log_laplacian(_as_backend_array(field, be), eps=eps)


def b_laplacian(field, eps: float = 1e-12, backend=None):
    """
    B-Laplacian: exp(Δ ln(field)).

    This is a multiplicative analogue of the usual Laplacian.
    """
    be = _backend(backend)
    lap_log = log_laplacian(field, eps=eps, backend=be)
    return be.exp(lap_log)


def divergence(Fy, Fx, backend=None):
    """
    Divergence of a 2D vector field F = (Fy, Fx) using central differences
    with periodic-like boundaries.
    """
    be = _backend(backend)
    return be.divergence(_as_backend_array(Fy, be), _as_backend_array(Fx, be))


# torch reworks

def log_gradient_torch(field: torch.Tensor, eps: float = 1e-12):
    """
    Torch version of log_gradient.
    field: 2D tensor (H, W)
    """
    import torch
    f = torch.clamp(field, min=eps)
    logf = torch.log(f)

    gy = 0.5 * (torch.roll(logf, shifts=-1, dims=0) - torch.roll(logf, shifts=1, dims=0))
    gx = 0.5 * (torch.roll(logf, shifts=-1, dims=1) - torch.roll(logf, shifts=1, dims=1))
    return gy, gx


def b_gradient_torch(field: torch.Tensor, eps: float = 1e-12):
    import torch
    gy, gx = log_gradient_torch(field, eps=eps)
    return torch.exp(gy), torch.exp(gx)


def log_laplacian_torch(field: torch.Tensor, eps: float = 1e-12):
    import torch
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


def b_laplacian_torch(field: torch.Tensor, eps: float = 1e-12):
    import torch
    lap_log = log_laplacian_torch(field, eps=eps)
    return torch.exp(lap_log)


def divergence_torch(Fy: torch.Tensor, Fx: torch.Tensor):
    import torch
    dFy_dy = 0.5 * (torch.roll(Fy, -1, 0) - torch.roll(Fy, 1, 0))
    dFx_dx = 0.5 * (torch.roll(Fx, -1, 1) - torch.roll(Fx, 1, 1))
    return dFy_dy + dFx_dx

__all__ = [
    "NUMPY_BACKEND",
    "TORCH_BACKEND",
    "set_backend",
    "BACKEND_NAME",
    "xp",
    "b_add",
    "b_sub",
    "b_mult",
    "b_div",
    "b_integral",
    "b_derivative",
    "b_derivative_torch",
    "b_integral_torch",
    "b_derivative_sympy",
    "log_gradient",
    "b_gradient",
    "log_laplacian",
    "b_laplacian",
    "divergence",
    "log_gradient_torch",
    "b_gradient_torch",
    "log_laplacian_torch",
    "b_laplacian_torch",
    "divergence_torch",
]
