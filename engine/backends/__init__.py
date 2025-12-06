# engine/backends/__init__.py
from importlib import import_module

def select_backend():
    """
    Returns a tuple of (backend_name, backend_module).

    Logic:
    1) Try to import torch.
    2) If torch imports and `torch.cuda.is_available()` is True, pick torch.
    3) Otherwise, fall back to NumPy.

    Notes:
    - Uses lazy import to avoid hard dependency on torch.
    - Keeps the API surface small: caller gets the module to use directly.
    """
    torch = None
    try:
        torch = import_module("torch")
    except Exception:
        torch = None

    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "torch", torch
        except Exception:
            # If CUDA check fails (rare), still allow CPU torch rather than crash
            return "torch", torch

    import numpy as np  # NumPy is assumed to be available
    return "numpy", np
