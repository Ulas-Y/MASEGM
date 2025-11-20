import numpy as np


class MatterField:
    """
    Represents a simple matter density field (no velocities yet).

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape (ny, nx).
    initial_value : float, optional
        Initial matter density in every cell.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.density = np.full(shape, initial_value, dtype=float)

    def add_matter(self, y: int, x: int, amount: float) -> None:
        self.density[y, x] += amount

    def remove_matter(self, y: int, x: int, amount: float) -> None:
        self.density[y, x] = max(0.0, self.density[y, x] - amount)

    def total_mass(self) -> float:
        return float(self.density.sum())

    def copy(self) -> "MatterField":
        mf = MatterField(self.shape)
        mf.density = self.density.copy()
        return mf
