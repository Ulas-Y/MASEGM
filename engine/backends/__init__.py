# engine/backends/__init__.py

"""
Backend factory utilities.

New backends can self-register by calling :func:`register_backend` at module
import time, e.g. ``register_backend("jax", JAXBackend)``.
"""

from typing import Callable, Dict

BackendFactory = Callable[..., object]

_backend_registry: Dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend factory.

    Future backend modules (e.g., JAX) can call this at module import time to
    make themselves discoverable by :func:`get_backend`.
    """

    _backend_registry[name.lower()] = factory


def get_backend(name: str = "numpy", **kwargs):
    """Return an instance of the requested backend (case-insensitive)."""

    factory = _backend_registry.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown backend: {name}")
    return factory(**kwargs)


def auto_backend(**kwargs):
    """Pick the best available backend lazily.

    Prefers the Torch backend when CUDA is available; otherwise falls back to
    NumPy. Import and CUDA checks are wrapped in try/except to avoid failing
    when torch is not installed or misconfigured.
    """

    torch = None
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None

    if torch is not None:
        try:
            if torch.cuda.is_available() and "torch" in _backend_registry:
                return _backend_registry["torch"](**kwargs)
        except Exception:
            # Fall back to NumPy on any CUDA probe failure.
            pass

    if "numpy" not in _backend_registry:
        raise RuntimeError("NumPy backend is not registered.")
    return _backend_registry["numpy"](**kwargs)


from .backends_numpy import NumpyBackend  # noqa: E402

register_backend("numpy", NumpyBackend)

try:  # noqa: E402
    from .backends_torch import TorchBackend
except ImportError:  # noqa: E402
    TorchBackend = None
else:
    register_backend("torch", TorchBackend)

__all__ = [
    "auto_backend",
    "get_backend",
    "register_backend",
    "NumpyBackend",
    "TorchBackend",
]
