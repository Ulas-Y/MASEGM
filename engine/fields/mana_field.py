import numpy as np

from engine.math.b_calculus import (
    b_add,
    b_mult,
    divergence,
    get_backend,
    laplacian,
    log_gradient,
    log_laplacian,
)



class ManaField:
    """
    Represents a mana density field on a 2D grid.
    """
    
    def __init__(self, shape=(100, 100), initial_value: float = 0.0):
        self.shape = shape
        backend = get_backend()
        base = max(initial_value, 0.0)
        self.grid = backend.full(shape, base) + backend.asarray(1e-12)

        # NEW: phase index per cell (will be managed by phase rules)
        # 0: particles, 1: energy, 2: gas, 3: refined, 4: aether, 5: purinium
        phase_dtype = np.uint8
        if backend.__class__.__name__ == "TorchBackend":
            import torch

            phase_dtype = torch.uint8

        self.phase = backend.zeros(shape, dtype=phase_dtype)
    
    def add_mana(self, y: int, x: int, amount: float) -> None:
        backend = get_backend()
        self.grid[y, x] += backend.asarray(amount)

    def remove_mana(self, y: int, x: int, amount: float) -> None:
        backend = get_backend()
        updated = self.grid[y, x] - backend.asarray(amount)
        self.grid[y, x] = backend.maximum(updated, backend.asarray(0.0))
    
    def total_mana(self) -> float:
        total = self.grid.sum()
        return float(total.item() if hasattr(total, "item") else total)
    
    def diffuse(self, rate: float, dt: float) -> None:
        """
        Simple explicit diffusion step:
        ∂M/∂t = rate * ∇² M
        
        This is VERY basic and not numerically perfect,
        but good for prototyping.
        """
        backend = get_backend()
        lap = laplacian(self.grid, backend=backend)
        rate_arr = backend.asarray(rate)
        dt_arr = backend.asarray(dt)
        self.grid = self.grid + rate_arr * dt_arr * lap
    
    def b_diffuse(self, rate: float, dt: float) -> None:
        """
        B-diffusion: multiplicative analogue of diffusion.

        In log-space:
            log m_new = log m + rate * Δ log m * dt
        So in normal space:
            m_new = m * exp(rate * Δ log m * dt)
        """
        backend = get_backend()
        lap_log = log_laplacian(self.grid, backend=backend)
        rate_arr = backend.asarray(rate)
        dt_arr = backend.asarray(dt)
        self.grid = self.grid * backend.exp(rate_arr * lap_log * dt_arr)
        self.ensure_positive()
    
    def b_advect(self, strength: float, dt: float) -> None:
        """
        B-style advection / transport of mana.

        We build a velocity field v from gradients of ln(m):

            gy, gx = grad ln m
            v = -strength * grad ln m

        Then evolve mana using a discrete continuity equation:

            ∂m/∂t + div(m v) = 0  → m_new = m_old - dt * div(m v)
        """
        if strength == 0.0:
            return

        # Use log-gradient so behaviour is multiplicative-scale aware
        backend = get_backend()
        gy, gx = log_gradient(self.grid, backend=backend)

        # Velocity field (you can flip the sign if you want flows from low→high)
        strength_arr = backend.asarray(strength)
        vy = -strength_arr * gy
        vx = -strength_arr * gx

        # Flux = m * v
        Fy = self.grid * vy
        Fx = self.grid * vx

        # Divergence of the flux
        divF = divergence(Fy, Fx, backend=backend)

        # Advection update
        dt_arr = backend.asarray(dt)
        self.grid = self.grid - dt_arr * divF
        self.ensure_positive()
    
    def copy(self) -> "ManaField":
        mf = ManaField(self.shape)
        mf.grid = self.grid.clone() if hasattr(self.grid, "clone") else self.grid.copy()
        mf.phase = self.phase.clone() if hasattr(self.phase, "clone") else self.phase.copy()
        return mf
    
    def ensure_positive(self, eps: float = 1e-12) -> None:
        """
        Ensure that all mana values stay strictly positive
        (needed for B-calculus, which uses ln).
        """
        backend = get_backend()
        self.grid = backend.maximum(self.grid, backend.asarray(eps))
    
    def b_scale_mult(self, factor: float) -> None:
        """
        Apply a multiplicative scaling on the B-scale:
        
            mana_new = mana ⊕ factor = mana * factor
        
        Using your B-add definition.
        """
        self.ensure_positive()
        backend = get_backend()
        self.grid = b_add(self.grid, backend.asarray(factor))  # which is just self.grid * factor
    
    def b_scale_power(self, exponent: float) -> None:
        """
        Apply B-multiplication-like behaviour:
        
            mana_new = mana ⊗ exponent = mana^exponent
        """
        self.ensure_positive()
        backend = get_backend()
        self.grid = b_mult(self.grid, backend.asarray(exponent))  # which is self.grid ** exponent
