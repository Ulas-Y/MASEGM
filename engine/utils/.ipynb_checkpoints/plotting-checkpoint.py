import matplotlib.pyplot as plt
import numpy as np

from .math_utils import normalize_field


def plot_scalar_field(field: np.ndarray, title: str = "Field") -> None:
    """
    Quick 2D visualization for a scalar field.
    """
    img = normalize_field(field)
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.colorbar(label="normalized value")
    plt.title(title)
    plt.tight_layout()
    plt.show()
