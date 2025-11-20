import numpy as np
from scipy.signal import convolve2d  # uses SciPy you installed


class ManaField:
    """
    Represents a mana density field on a 2D grid.
    """

    _LAPLACIAN_KERNEL = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]]
    )

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, initial_value, dtype=float)

    def add_mana(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] += amount

    def remove_mana(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] = max(0.0, self.grid[y, x] - amount)

    def total_mana(self) -> float:
        return float(self.grid.sum())

    def diffuse(self, rate: float, dt: float) -> None:
        """
        Simple explicit diffusion step:
        ∂M/∂t = rate * ∇² M

        This is VERY basic and not numerically perfect,
        but good for prototyping.
        """
        lap = convolve2d(self.grid, self._LAPLACIAN_KERNEL, mode="same", boundary="symm")
        self.grid += rate * dt * lap

    def copy(self) -> "ManaField":
        mf = ManaField(self.shape)
        mf.grid = self.grid.copy()
        return mf
