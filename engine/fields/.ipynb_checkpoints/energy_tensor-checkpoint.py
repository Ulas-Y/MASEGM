import numpy as np
from engine.math.b_calculus import log_laplacian


class EnergyTensor:
    """
    Simple scalar energy field.
    Compatible with .grid like ManaField and MatterField.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, initial_value, dtype=float)

    def ensure_nonnegative(self) -> None:
        self.grid = np.maximum(self.grid, 0.0)

    def add_energy(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] += amount
        self.ensure_nonnegative()

    def total_energy(self) -> float:
        return float(self.grid.sum())

    def b_diffuse(self, rate: float, dt: float) -> None:
        """
        B-diffusion for energy, analogous to mana.b_diffuse.
        """
        lap_log = log_laplacian(self.grid)
        self.grid *= np.exp(rate * lap_log * dt)
        self.ensure_nonnegative()
