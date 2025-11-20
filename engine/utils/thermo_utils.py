import numpy as np


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
    total = field.sum()
    if total <= 0:
        return 0.0

    p = field / total
    p = np.maximum(p, eps)      # avoid log(0)
    s = -np.sum(p * np.log(p))
    return float(s)

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
