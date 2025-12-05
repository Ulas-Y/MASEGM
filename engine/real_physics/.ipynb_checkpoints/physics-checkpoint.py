import numpy as np
from engine.fields.mana_field import ManaField
from engine.math.b_calculus import log_gradient  # For efficient gradients

class PhysicsRule:
    """Base for physics ops on fields."""
    def apply(self, mana: ManaField, dt: float) -> None:
        raise NotImplementedError

class GravityAttraction(PhysicsRule):
    """Simple gravity: Mana attracts mana via gradient-based force."""
    def __init__(self, G: float = 0.1):  # Gravity constant, tune low to avoid blowups
        self.G = G

    def apply(self, mana: ManaField, dt: float) -> None:
        # Compute log-gradient for B-efficiency (directional pull to dense areas)
        grad_y, grad_x = log_gradient(mana.grid)
        force_y = -self.G * grad_y  # Toward higher density
        force_x = -self.G * grad_x
        # Advect mana along force (simple Euler step)
        mana.advect(force_y, force_x, dt)  # Use your existing advect method

# Add more, e.g., VelocityFieldRule for momentum if needed