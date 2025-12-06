# engine/backends/__init__.py

"""
Backend factory utilities.

New backends can self-register by calling :func:`register_backend` at module
import time, e.g. ``register_backend("jax", JAXBackend)``.
"""

from typing import Callable, Dict, Optional

BackendFactory = Callable[..., object]

_backend_registry: Dict[str, BackendFactory] = {}
TorchBackend: Optional[BackendFactory] = None


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend factory.

    Future backend modules (e.g., JAX) can call this at module import time to
    make themselves discoverable by :func:`get_backend`.
    """

    _backend_registry[name.lower()] = factory


def _ensure_torch_registered() -> bool:
    """Attempt to import and register the Torch backend lazily."""

    global TorchBackend

    if TorchBackend is not None and "torch" in _backend_registry:
        return True

    try:
        from .backends_torch import TorchBackend as _TorchBackend  # noqa: E402
    except Exception:
        return False

    TorchBackend = _TorchBackend
    register_backend("torch", TorchBackend)
    return True


def get_backend(name: str = "numpy", **kwargs):
    """Return an instance of the requested backend (case-insensitive)."""

    normalized = name.lower()
    if normalized == "torch":
        _ensure_torch_registered()

    factory = _backend_registry.get(normalized)
    if factory is None:
        available = ", ".join(sorted(_backend_registry)) or "none"
        raise ValueError(f"Unknown backend: {name}. Available: {available}")
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
    except Exception:
        torch = None

    if torch is not None:
        try:
            if torch.cuda.is_available() and _ensure_torch_registered():
                return get_backend("torch", **kwargs)
        except Exception:
            # Fall back to NumPy on any CUDA probe failure.
            pass

    return get_backend("numpy", **kwargs)


from .backends_numpy import NumpyBackend  # noqa: E402

register_backend("numpy", NumpyBackend)

__all__ = [
    "auto_backend",
    "get_backend",
    "register_backend",
    "NumpyBackend",
    "TorchBackend",
]
