import numpy as np
from dataclasses import dataclass


def clamp(x, xmin=None, xmax=None):
    """Clamp a value or array between xmin and xmax."""
    if xmin is not None:
        x = np.maximum(x, xmin)
    if xmax is not None:
        x = np.minimum(x, xmax)
    return x


def normalize_field(field: np.ndarray) -> np.ndarray:
    """Return a normalized copy of the field (0..1)."""
    fmin = field.min()
    fmax = field.max()
    if fmax == fmin:
        return np.zeros_like(field)
    return (field - fmin) / (fmax - fmin)


@dataclass(frozen=True)
class BScaleValue:
    """
    Represents a value on your custom B-scale.

    Conceptually, this is the 'n' in 'nB'.

    Right now, we implement a *simple* model you can tweak:

        - For n > 0: factor(n) = n
        - For n < 0: factor(n) = 1 / (-n)

    So:
        2B   -> factor 2
        -2B  -> factor 1/2

    This matches your examples like 2B = 2x and -2B = 1/2 x.

    TODO:
        Replace `to_factor` and `from_factor` with your exact formulas
        when you finalize the B-scale math.
    """
    n: float

    def to_factor(self) -> float:
        """
        Map this B-scale value to a multiplicative factor.

        You can change this mapping later without changing the rest of the code.
        """
        if self.n > 0:
            return self.n
        elif self.n < 0:
            return 1.0 / (-self.n)
        else:
            # 0B is special / undefined in your system;
            # here we just return 1.0 as a neutral element.
            return 1.0

    @staticmethod
    def from_factor(factor: float) -> "BScaleValue":
        """
        Inverse mapping: from multiplicative factor back to a B-scale n.

        Current placeholder:
            - if factor >= 1:    n = factor
            - if 0 < factor < 1: n = -1/factor

        Modify this to fit your exact axioms later.
        """
        if factor >= 1.0:
            n = factor
        else:
            n = -1.0 / factor
        return BScaleValue(n)

    # Some convenient operators:

    def __mul__(self, other):
        """
        Combine two B-scale values by multiplying their factors.

        This is equivalent to applying both scalings.
        """
        if not isinstance(other, BScaleValue):
            return NotImplemented
        f = self.to_factor() * other.to_factor()
        return BScaleValue.from_factor(f)

    def __truediv__(self, other):
        if not isinstance(other, BScaleValue):
            return NotImplemented
        f = self.to_factor() / other.to_factor()
        return BScaleValue.from_factor(f)

    def apply_to(self, x: float) -> float:
        """
        Apply this B-scale value to an x (i.e. scale x).

        Example:
            BScaleValue(2).apply_to(x) = 2 * x
            BScaleValue(-2).apply_to(x) = (1/2) * x
        """
        return self.to_factor() * x
