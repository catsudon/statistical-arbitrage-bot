from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.risk import (
    aggregate_weights,
    apply_position_limits,
    clamp_gross_leverage,
    drawdown_curve,
    halt_after_drawdown,
    kelly_fraction,
    vol_target_scale,
    RiskLimits,
)


def test_vol_target_scale_clipped():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 1000))
    scale = vol_target_scale(rets, target_annual_vol=0.15, timeframe="1h", lookback=200)
    assert (scale >= 0).all() and (scale <= 5.0).all()


def test_kelly_bounded():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    k = kelly_fraction(rets)
    assert -1.0 <= k <= 1.0


def test_clamp_gross_leverage():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    w = pd.DataFrame({"A": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      "B": [2, 2, 2, 2, 2, 0, 0, 0, 0, 0]}, index=idx, dtype=float)
    cl = clamp_gross_leverage(w, max_gross=2.0)
    assert (cl.abs().sum(axis=1) <= 2.0 + 1e-9).all()


def test_drawdown_and_halt():
    eq = pd.Series([100, 110, 105, 95, 80, 85])
    dd = drawdown_curve(eq)
    assert dd.iloc[-2] < -0.25  # peaked at 110, fell to 80 -> -0.273
    halt = halt_after_drawdown(eq, max_dd=0.20)
    assert not halt.iloc[-1]


def test_aggregate_weights_sums():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    a = pd.DataFrame({"X": [1, 1, 1], "Y": [0, 0, 0]}, index=idx, dtype=float)
    b = pd.DataFrame({"X": [-0.5, -0.5, -0.5], "Y": [0.5, 0.5, 0.5]}, index=idx, dtype=float)
    agg = aggregate_weights({"a": a, "b": b})
    assert (agg["X"] == 0.5).all()
    assert (agg["Y"] == 0.5).all()


def test_apply_position_limits_caps_count():
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    w = pd.DataFrame({c: [0.4, 0.4] for c in "ABCDE"}, index=idx, dtype=float)
    lim = RiskLimits(max_positions=3, max_single_weight=1.0)
    out = apply_position_limits(w, lim)
    assert (out != 0).sum(axis=1).max() == 3
