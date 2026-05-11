"""Position sizing: vol targeting, fractional Kelly, ATR sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_PERIODS = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520, "1h": 8760, "4h": 2190, "1d": 365}


def annualize_factor(timeframe: str) -> float:
    return float(TRADING_PERIODS.get(timeframe, 252))


def vol_target_scale(
    returns: pd.Series,
    target_annual_vol: float,
    timeframe: str = "1h",
    lookback: int = 240,
    cap: float = 5.0,
) -> pd.Series:
    """Per-bar scaling factor that targets `target_annual_vol`."""
    af = annualize_factor(timeframe)
    realized = returns.rolling(lookback, min_periods=max(20, lookback // 4)).std() * np.sqrt(af)
    scale = (target_annual_vol / realized).clip(upper=cap).fillna(0.0)
    return scale.rename("vol_scale")


def kelly_fraction(returns: pd.Series, fraction: float = 0.25) -> float:
    """Single-asset Kelly (fractional): μ / σ². Returns a clipped fraction."""
    r = returns.dropna()
    if len(r) < 30:
        return 0.0
    mu = r.mean()
    var = r.var(ddof=1)
    if var <= 0:
        return 0.0
    k = mu / var
    return float(max(-1.0, min(1.0, fraction * k)))


def atr_position_size(price: float, atr: float, capital: float, risk_per_trade: float) -> float:
    """Risk-parity sizing: position = capital * risk / ATR. Returns qty."""
    if atr <= 0 or price <= 0:
        return 0.0
    dollar_at_risk = capital * risk_per_trade
    return float(dollar_at_risk / atr)
