import numpy as np


def clamp(x, xmin=None, xmax=None):
    """Clamp a value or array between xmin and xmax."""
    if xmin is not None:
        x = np.maximum(x, xmin)
    if xmax is not None:
        x = np.minimum(x, xmax)
    return x


def normalize_field(field: np.ndarray) -> np.ndarray:
    """Return a normalized copy of the field (0..1)."""
    fmin = field.min()
    fmax = field.max()
    if fmax == fmin:
        return np.zeros_like(field)
    return (field - fmin) / (fmax - fmin)
