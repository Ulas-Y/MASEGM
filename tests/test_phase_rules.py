import pathlib
import sys
import types

import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

if torch is None:
    pytest.skip("torch backend not available", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Minimal numpy/scipy stubs so we can import engine modules without the
# real dependencies installed in the execution environment. These rely on
# torch tensors underneath, which is sufficient for the torch-only test
# below.
if torch is not None and "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    def _tensor(x, dtype=None):
        return torch.tensor(x, dtype=dtype or torch.float64)

    numpy_stub.array = _tensor
    numpy_stub.asarray = _tensor
    numpy_stub.full = lambda shape, fill_value, dtype=float: torch.full(
        shape, fill_value, dtype=torch.float64 if dtype is float else dtype
    )
    numpy_stub.zeros = lambda shape, dtype=float: torch.zeros(
        shape, dtype=torch.float64 if dtype is float else dtype
    )
    numpy_stub.zeros_like = torch.zeros_like
    numpy_stub.maximum = torch.maximum
    numpy_stub.exp = torch.exp
    numpy_stub.clip = torch.clamp
    numpy_stub.float64 = torch.float64
    numpy_stub.uint8 = torch.uint8
    numpy_stub.ndarray = torch.Tensor

    sys.modules["numpy"] = numpy_stub

if torch is not None and "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    signal_stub = types.ModuleType("scipy.signal")

    def _convolve2d(*args, **kwargs):  # pragma: no cover - not used in test
        raise NotImplementedError

    signal_stub.convolve2d = _convolve2d
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.signal"] = signal_stub

from engine.rules.phase_rules import PhaseTransitionRule
from engine.physics.mana_phase import PhaseCode


@pytest.mark.skipif(torch is None, reason="torch backend not available")
def test_entropy_feedback_uses_backend_zero():
    rule = PhaseTransitionRule()
    dt = 0.1

    mana_grid = torch.ones((2, 2), dtype=torch.float32)
    purity = torch.tensor(
        [[0.95, 0.91], [0.92, 0.85]],
        dtype=torch.float32,
    )
    phase = torch.tensor(
        [
            [int(PhaseCode.AETHER), int(PhaseCode.AETHER)],
            [int(PhaseCode.PURINIUM), int(PhaseCode.GAS)],
        ],
        dtype=torch.int64,
    )

    initial = mana_grid.clone()
    rule._apply_entropy_feedback(mana_grid, purity, phase, dt)

    delta_p = purity - rule.thresholds.p_pivot
    expected = initial.clone()
    for code in PhaseCode:
        props = rule._phase_props[int(code)]
        mask = phase == int(code)
        if not torch.any(mask):
            continue
        local_delta = delta_p[mask]
        local = props.entropy_feedback * local_delta
        if props.purify_boost != 0.0:
            zero = torch.zeros_like(local_delta)
            local += props.purify_boost * torch.maximum(local_delta, zero)
        if props.mana_decay != 0.0:
            local -= props.mana_decay
        factor = torch.clamp(1.0 + local * dt, 0.0, 1e6)
        expected[mask] = initial[mask] * factor

    assert torch.allclose(mana_grid, expected)
