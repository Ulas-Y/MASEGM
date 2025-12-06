# engine/backends/__init__.py
from importlib import import_module
from importlib.util import find_spec

from .backends_numpy import NumpyBackend


def select_backend():
    """
    Returns a tuple of (backend_name, backend_instance).

    Logic:
    1) If torch is installed, return a TorchBackend (preferring CUDA when
       available, otherwise CPU).
    2) Otherwise, fall back to a NumPy backend.

    Notes:
    - ``Backend`` instances expose a consistent API (log/exp/roll/etc.) so
      callers can be backend-agnostic.
    - ``find_spec`` is used instead of try/except around imports to respect the
      codebase's no-try/except-on-import guideline.
    """

    if find_spec("torch") is not None:
        torch = import_module("torch")
        from .backends_torch import TorchBackend

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        return "torch", TorchBackend(device=device, dtype=torch.float64)

    return "numpy", NumpyBackend()
