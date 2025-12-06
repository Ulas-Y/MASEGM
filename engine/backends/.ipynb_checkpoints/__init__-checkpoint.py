# engine/backends/__init__.py

from .backend_numpy import NumpyBackend

try:
    from .backend_torch import TorchBackend
except ImportError:
    TorchBackend = None


def get_backend(name="numpy", **kwargs):
    name = name.lower()
    if name == "numpy":
        return NumpyBackend(**kwargs)
    elif name == "torch":
        if TorchBackend is None:
            raise RuntimeError("Torch backend requested but torch is not installed.")
        return TorchBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")
