from dataclasses import dataclass
from typing import Any

from engine.math import b_calculus

xp = b_calculus.xp


def clamp(x: Any, xmin=None, xmax=None):
    """Clamp a value or array between xmin and xmax using the active backend."""

    x = xp.asarray(x)
    if xmin is not None and xmax is not None:
        return xp.clip(x, xmin, xmax)
    if xmin is not None:
        return xp.maximum(x, xmin)
    if xmax is not None:
        return xp.clip(x, float("-inf"), xmax)
    return x


def normalize_field(field: Any) -> Any:
    """Return a normalized copy of the field (0..1) on the active backend."""

    arr = xp.asarray(field)
    fmin = arr.min()
    fmax = arr.max()
    denom = fmax - fmin

    if denom.item() == 0:  # type: ignore[attr-defined]
        return xp.zeros(arr.shape, dtype=getattr(arr, "dtype", None))

    return (arr - fmin) / denom


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
