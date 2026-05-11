"""Trading-cost models. Costs returned are *as a fraction of notional*."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    fee_bps: float = 4.0           # taker fee, one-way
    slippage_bps: float = 2.0      # per turnover
    funding_bps_per_day: float = 1.0  # for perpetuals; ignored for spot
    include_funding: bool = False

    def trade_cost(self, turnover_fraction: float) -> float:
        """Cost of trading `turnover_fraction` of NAV."""
        per_side = (self.fee_bps + self.slippage_bps) / 10_000.0
        return per_side * turnover_fraction

    def funding_cost(self, gross_exposure: float, bars_held: float, bars_per_day: float) -> float:
        if not self.include_funding:
            return 0.0
        days = bars_held / bars_per_day
        return (self.funding_bps_per_day / 10_000.0) * gross_exposure * days
