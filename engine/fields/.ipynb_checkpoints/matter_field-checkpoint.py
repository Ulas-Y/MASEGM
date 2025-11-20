import numpy as np


class MatterField:
    """
    Represents a matter density field on a 2D grid.
    Mirrors the structure of ManaField for consistency.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, initial_value, dtype=float)

    def total_matter(self) -> float:
        return float(self.grid.sum())

    def add_matter(self, amount: float) -> None:
        self.grid += amount

    def remove_matter(self, amount: float) -> None:
        self.grid -= amount
        self.grid = np.maximum(self.grid, 0.0)
