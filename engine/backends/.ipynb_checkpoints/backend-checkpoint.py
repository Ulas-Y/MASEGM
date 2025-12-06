# engine/backends/backend.py

from abc import ABC, abstractmethod
from typing import Any, Tuple


class Backend(ABC):
    """Abstract base class for NumPy/Torch backends.

    Concrete implementations wrap either NumPy or Torch and must mirror
    semantics between the two libraries (e.g., broadcasting, dtype casting,
    and device placement) so the surrounding code can be agnostic to the
    numerical engine in use.
    """

    # ---- basic array ops ----
    @abstractmethod
    def asarray(self, x: Any) -> Any:
        """Convert ``x`` to a backend-native floating array/tensor.

        Implementations should:

        * Cast to the backend's default floating dtype while preserving the
          device (for Torch) or returning a NumPy array with that dtype.
        * Accept any sequence, scalar, or existing array/tensor and apply the
          library's standard broadcasting rules when combined with other
          operands.
        * Avoid copying when the input already matches the target dtype and
          device unless required for type safety.
        """

    @abstractmethod
    def log(self, x: Any) -> Any:
        """Compute the natural logarithm elementwise.

        Inputs should be non-negative and preferably strictly positive; callers
        are expected to clamp inputs to a minimum ``eps`` (≥1e-12) before
        calling to prevent ``-inf`` values. The output must retain the dtype
        and device of ``x`` and follow backend broadcasting behavior.
        """

    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Compute the exponential elementwise.

        The output should use the same dtype/device as ``x``. Implementations
        should rely on backend-provided numerically stable exponentials and
        allow broadcasting of ``x`` with other operands.
        """

    @abstractmethod
    def maximum(self, x: Any, val: float) -> Any:
        """Elementwise maximum between ``x`` and scalar ``val``.

        This is typically used to enforce lower bounds (e.g., stability eps).
        Implementations must respect broadcasting rules and ensure the output
        lives on the same device/dtype as ``x``.
        """

    @abstractmethod
    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        """Clip ``x`` to the inclusive range ``[min_val, max_val]``.

        Used to bound quantities before logarithms or exponentials; callers may
        pass ``min_val`` of at least 1e-12 to avoid numerical underflow.
        Implementations should preserve dtype/device and follow backend
        clipping semantics, including broadcasting of the scalar bounds.
        """

    # ---- B-arithmetic ----
    @abstractmethod
    def b_add(self, x: Any, y: Any) -> Any:
        """B-addition (multiplication) defined as ``x * y``.

        B-operations operate in a multiplicative semiring where multiplication
        represents addition in log-space. Inputs must be broadcastable to a
        common shape under NumPy/Torch rules, with outputs retaining the common
        dtype/device.
        """

    @abstractmethod
    def b_sub(self, x: Any, y: Any) -> Any:
        """B-subtraction (division) defined as ``x / y``.

        The operation mirrors subtraction in log-space. Implementations should
        rely on backend division semantics, using broadcasting and preserving
        dtype/device. Callers are responsible for ensuring ``y`` is nonzero or
        suitably clamped before invocation.
        """

    @abstractmethod
    def b_mul(self, x: Any, y: Any) -> Any:
        """B-multiplication (power) defined as ``x ** y``.

        Conceptually corresponds to scaling in log-space. Inputs must be
        broadcastable and use backend-native dtype/device; implementations
        should leverage built-in power functions and avoid casting away the
        configured precision.
        """

    @abstractmethod
    def b_div(self, x: Any, y: Any, eps: float = 1e-30) -> Any:
        """B-division defined as ``log(x) / log(y)`` with stability guards.

        Implementations should clamp ``x`` to at least ``eps`` (default
        1e-30) and replace non-positive or ``≈1`` values of ``y`` with ``eps``
        to avoid divide-by-zero or log-of-one instabilities. All operands must
        share dtype/device after broadcasting to a common shape.
        """

    # ---- grid operators ----
    @abstractmethod
    def log_gradient(self, field: Any, eps: float = 1e-12) -> Tuple[Any, Any]:
        """Compute central differences of ``log(field)`` along ``y`` and ``x``.

        ``field`` is expected to be a 2D array/tensor ``(H, W)`` of positive
        values. Implementations must clamp to ``eps`` (≥1e-12) before applying
        ``log`` to avoid ``-inf``. The returned tuple ``(gy, gx)`` matches the
        input shape and dtype/device. Rolling/periodic boundary behavior should
        be consistent between backends.
        """

    @abstractmethod
    def log_laplacian(self, field: Any, eps: float = 1e-12) -> Any:
        """Compute the discrete Laplacian of ``log(field)`` on a 2D grid.

        ``field`` should be shape ``(H, W)`` with positive entries. Values are
        clamped to ``eps`` (≥1e-12) prior to ``log``. The operator uses periodic
        boundary conditions (via ``roll``) and must mirror NumPy/Torch
        semantics for indexing, broadcasting, dtype, and device placement.
        """

    @abstractmethod
    def divergence(self, Fy: Any, Fx: Any) -> Any:
        """Compute divergence of a 2D vector field ``(Fy, Fx)``.

        ``Fy`` and ``Fx`` should be arrays/tensors with matching shapes
        ``(H, W)`` and compatible dtype/device. Central differences are used
        along the corresponding axes with periodic boundaries. Broadcasting of
        inputs should follow backend rules, and outputs must maintain the
        common dtype/device.
        """
