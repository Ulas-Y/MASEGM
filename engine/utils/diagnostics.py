# engine/utils/thermo_utils.py  (or diagnostics.py)

import numpy as np

from engine.math import b_calculus

PHASE_NAMES = ("particles", "plasma", "gas", "liquid", "aether", "purinium")


def phase_histogram(phase: np.ndarray) -> dict:
    """
    Count how many cells are in each phase code 0..5.
    Returns a dict {name: count}.
    """
    be = b_calculus.xp
    phase_be = be.asarray(phase)

    total = phase_be.numel() if hasattr(phase_be, "numel") else phase_be.size

    counts = {}
    for code, name in enumerate(PHASE_NAMES):
        mask = phase_be == be.asarray(code)
        n = mask.sum()
        if hasattr(n, "item"):
            n = n.item()
        counts[name] = int(n)

    counts["total_cells"] = int(total.item() if hasattr(total, "item") else total)
    return counts


def print_phase_stats(phase: np.ndarray) -> None:
    """
    Pretty-print phase counts + percentages.
    """
    stats = phase_histogram(phase)
    total = stats.pop("total_cells")
    for name, n in stats.items():
        frac = n / total if total > 0 else 0.0
        print(f"{name:9s}: {n:6d}  ({frac*100:5.1f}%)")
    print(f"total   : {total}")
