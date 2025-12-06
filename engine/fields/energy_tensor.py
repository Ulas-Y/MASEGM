from engine.math.b_calculus import log_laplacian, xp


class EnergyTensor:
    """
    Simple scalar energy field.
    Compatible with .grid like ManaField and MatterField.
    """

    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        if hasattr(xp, "full"):
            self.grid = xp.full(shape, initial_value)
        else:
            self.grid = xp.asarray([[initial_value for _ in range(shape[1])] for _ in range(shape[0])])

    def ensure_nonnegative(self) -> None:
        self.grid = xp.maximum(self.grid, xp.asarray(0.0))

    def add_energy(self, y: int, x: int, amount: float) -> None:
        self.grid[y, x] += xp.asarray(amount)
        self.ensure_nonnegative()

    def total_energy(self) -> float:
        total = self.grid.sum()
        return float(total.item() if hasattr(total, "item") else total)

    def b_diffuse(self, rate: float, dt: float) -> None:
        """
        B-diffusion for energy, analogous to mana.b_diffuse.
        """
        lap_log = log_laplacian(self.grid)
        rate_arr = xp.asarray(rate)
        dt_arr = xp.asarray(dt)
        self.grid *= xp.exp(rate_arr * lap_log * dt_arr)
        self.ensure_nonnegative()
