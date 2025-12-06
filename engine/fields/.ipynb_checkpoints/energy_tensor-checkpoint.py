import numpy as np
from engine.math.b_calculus import log_laplacian, log_gradient, divergence  # For b_diffuse/b_advect

class EnergyTensor:
    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        self.grid = np.full(shape, max(initial_value, 0.0), dtype=float) + 1e-12

    def total_energy(self) -> float:
        return float(self.grid.sum())

    def b_diffuse(self, rate: float, dt: float) -> None:
        lap_log = log_laplacian(self.grid)
        self.grid *= np.exp(rate * lap_log * dt)
        self.ensure_nonnegative()

    def b_advect(self, strength: float, dt: float) -> None:
        # Same as ManaField's: v from log_grad, update via -dt * div(grid * v)
        if strength == 0.0:
            return
        gy, gx = log_gradient(self.grid)
        vy = -strength * gy
        vx = -strength * gx
        Fy = self.grid * vy
        Fx = self.grid * vx
        divF = divergence(Fy, Fx)
        self.grid -= dt * divF
        self.ensure_nonnegative()

    def ensure_nonnegative(self, eps: float = 1e-12) -> None:
        self.grid = np.maximum(self.grid, eps)