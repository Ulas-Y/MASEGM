"""
b_calculus.py

Full implementation of the B-Scale arithmetic, B-derivative, and B-integral
based on your formal definitions.

Author: YOU (concept) + ChatGPT (implementation)
"""

import numpy as np

# ============================================================
#  B-ARITHMETIC (Fundamental Operations)
# ============================================================

def b_add(x, y):
    """B-addition: x ⊕ y = x * y"""
    return x * y


def b_sub(x, y):
    """B-subtraction: x ⊖ y = x / y"""
    return x / y


def b_mul(x, y):
    """B-multiplication: x ⊗ y = x^y"""
    return x ** y


def b_div(x, y):
    """B-division: x ⊘ y = log_y(x)"""
    return np.log(x) / np.log(y)


# ============================================================
#  B-DERIVATIVE
# ============================================================

def b_derivative(f, x, dx=1e-8):
    """
    Computes D_B f(x) = exp( d/d(ln x) ln f(x) )

    Uses finite differences:
        d/d(ln x) g = (g(x+dx) - g(x-dx)) / ( ln(x+dx) - ln(x-dx) )
    """
    x = np.asarray(x)

    # avoid hitting zero
    xp = x + dx
    xm = x - dx
    xm = np.where(xm <= 0, x, xm)  # ensure positive

    g = lambda z: np.log(f(z))

    num = g(xp) - g(xm)
    den = np.log(xp) - np.log(xm)

    return np.exp(num / den)


# ============================================================
#  B-INTEGRAL (Continuous, Numeric Approximation)
# ============================================================

def b_integral(f, a, b, n=10000):
    """
    Compute ∫ f(x) d_B x = exp( ∫ ln(f(x)) d(ln x) )

    Numeric approach:
        I = exp( sum ln(f(x_i)) * d(ln x_i) )

    Parameters:
        f : function
        a,b : integration limits (>0)
        n : number of steps
    """
    if a <= 0 or b <= 0:
        raise ValueError("B-integral domain requires a>0, b>0.")

    xs = np.linspace(a, b, n)
    lnf = np.log(f(xs))
    dlnx = np.diff(np.log(xs))  # differences in ln(x)

    # midpoint rule: average ln(f) over intervals
    mid = (lnf[1:] + lnf[:-1]) / 2

    I = np.exp(np.sum(mid * dlnx))
    return I


# ============================================================
#  OPTIONAL: TORCH VERSION (GPU-READY)
# ============================================================

def b_derivative_torch(f, x, dx=1e-6):
    """
    Same as b_derivative but using torch for GPU acceleration.
    """
    import torch

    xp = x + dx
    xm = x - dx
    xm = torch.where(xm <= 0, x, xm)

    g = lambda z: torch.log(f(z))

    num = g(xp) - g(xm)
    den = torch.log(xp) - torch.log(xm)

    return torch.exp(num / den)


def b_integral_torch(f, a, b, n=20000):
    """
    GPU-accelerated B-integral.
    """
    import torch

    xs = torch.linspace(a, b, n, device="cuda")
    lnf = torch.log(f(xs))
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


