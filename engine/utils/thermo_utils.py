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
