import numpy as np


class ManaField:
    """
    Represents a mana density field on a 2D grid.

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape (ny, nx).
    initial_value : float, optional
        Initial mana in every cell.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, initial_value, dtype=float)

    def add_mana(self, y: int, x: int, amount: float) -> None:
        """Add mana at (y, x)."""
        self.grid[y, x] += amount

    def remove_mana(self, y: int, x: int, amount: float) -> None:
        """Remove mana at (y, x), not letting it go below zero."""
        self.grid[y, x] = max(0.0, self.grid[y, x] - amount)

    def total_mana(self) -> float:
        """Return total mana in the field."""
        return float(self.grid.sum())

    def copy(self) -> "ManaField":
        """Return a shallow copy of this field."""
        mf = ManaField(self.shape)
        mf.grid = self.grid.copy()
        return mf
