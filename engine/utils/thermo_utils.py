import numpy as np

from engine.math import b_calculus


def mana_entropy(field: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute global Shannon entropy of a mana-like scalar field.

    Parameters
    ----------
    field : np.ndarray
        Non-negative scalar field (e.g. mana.grid).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Entropy S = -sum p_ij log p_ij  (natural log).
    """
    be = b_calculus.xp

    field_be = be.asarray(field)
    total = field_be.sum()
    total_value = float(total.item() if hasattr(total, "item") else total)

    if total_value <= 0:
        return 0.0

    p = field_be / total
    p = be.maximum(p, be.asarray(eps))  # avoid log(0)
    s = -(p * be.log(p)).sum()

    if hasattr(be, "asnumpy"):
        return float(be.asnumpy(s))

    return float(s.item() if hasattr(s, "item") else s)

def detect_ness(series, window: int = 20, rtol: float = 1e-3, atol: float = 1e-6):
    """
    Simple NESS detector on a time series (total mana, entropy, etc.).

    Returns (is_ness, span, mean).

    is_ness = True if, over the last `window` points, the variation is small
    relative to the mean.
    """
    import numpy as np

    if len(series) < window:
        return False, None, None

    tail = np.array(series[-window:])
    span = float(tail.max() - tail.min())
    mean = float(tail.mean())
    scale = max(abs(mean), atol)

    is_ness = span <= max(rtol * scale, atol)
    return is_ness, span, mean
