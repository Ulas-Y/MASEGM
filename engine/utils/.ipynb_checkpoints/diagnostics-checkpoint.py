# engine/utils/thermo_utils.py  (or diagnostics.py)

import numpy as np

PHASE_NAMES = ("particles", "plasma", "gas", "liquid", "aether", "purinium")


def phase_histogram(phase: np.ndarray) -> dict:
    """
    Count how many cells are in each phase code 0..5.
    Returns a dict {name: count}.
    """
    counts = {}
    total = phase.size
    for code, name in enumerate(PHASE_NAMES):
        n = int((phase == code).sum())
        counts[name] = n
    counts["total_cells"] = int(total)
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
