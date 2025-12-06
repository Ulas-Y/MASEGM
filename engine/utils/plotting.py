# engine/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from engine.math import b_calculus
# ... existing plot_scalar_field here ...


def _normalize_numpy(field_np: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array to 0..1 without leaving NumPy space."""

    fmin = float(np.min(field_np))
    fmax = float(np.max(field_np))
    denom = fmax - fmin

    if denom == 0:
        return np.zeros_like(field_np, dtype=float)

    return (field_np - fmin) / denom


def plot_scalar_field(field: np.ndarray, title: str = "Field") -> None:
    """
    Quick 2D visualization for a scalar field.
    """
    be = b_calculus.get_backend()
    field_np = np.asarray(be.asnumpy(field))
    img = _normalize_numpy(field_np)

    plt.imshow(np.asarray(img), origin="lower", interpolation="nearest")
    plt.colorbar(label="normalized value")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_phase_map(phase: np.ndarray, title: str = "Mana phases") -> None:
    """
    Show integer phase map (0..5) with a discrete colormap and legend.
    Expects the phase codes from PhaseTransitionRule:
      0=particles, 1=plasma, 2=gas, 3=liquid, 4=aether, 5=purinium
    """
    # small discrete colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [
        "#444444",  # 0 particles
        "#ff8800",  # 1 plasma
        "#00aaff",  # 2 gas
        "#00cc55",  # 3 liquid
        "#aa00ff",  # 4 aether
        "#ffff00",  # 5 purinium
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 6.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    be = b_calculus.get_backend()
    phase_np = np.asarray(be.asnumpy(phase))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.asarray(phase_np), origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title)

    # make a tiny custom legend
    labels = [
        "0 particles",
        "1 plasma",
        "2 gas",
        "3 liquid",
        "4 aether",
        "5 purinium",
    ]
    # 6 tiny patches
    from matplotlib.patches import Patch
    patches = [Patch(color=colors[i], label=labels[i]) for i in range(6)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_purity_field(purity: np.ndarray, title: str = "Mana purity") -> None:
    """
    Continuous purity heatmap (0..1).
    """
    be = b_calculus.get_backend()
    purity_np = np.asarray(be.asnumpy(purity))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.asarray(purity_np), origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("purity (mana / (mana + matter))")
    plt.tight_layout()
    plt.show()

