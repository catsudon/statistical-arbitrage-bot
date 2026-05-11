"""End-to-end programmatic example.

1. fetch a small universe
2. scan for cointegrated pairs
3. backtest the best pair (vectorized) and the same with the event-driven engine
4. print metrics side-by-side
"""
from __future__ import annotations

import pandas as pd

from statarb.backtest import CostModel, event_backtest, vectorized_backtest
from statarb.core import PairSpec
from statarb.data import CCXTProvider, panel_closes
from statarb.scanner import PairsScanConfig, scan_pairs
from statarb.signals import ZScoreParams
from statarb.strategies import PairsStrategy, PairStrategyConfig
from statarb.universe import filter_universe


def main():
    provider = CCXTProvider(exchange="binance")
    symbols = provider.top_by_volume(n=20, quote="USDT")
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=180)
    closes = panel_closes(provider, symbols, "1h", start=start, end=end)
    universe = filter_universe(closes, min_history_bars=1500,
                               exclude=["BUSD/USDT", "USDC/USDT", "FDUSD/USDT"])
    closes = closes[universe]
    print(f"universe: {len(closes.columns)} symbols, {len(closes)} bars")

    pairs = scan_pairs(closes, PairsScanConfig(pvalue_max=0.05, workers=4))
    if not pairs:
        print("no pairs found")
        return
    best = pairs[0]
    print(f"best pair: {best.y} vs {best.x}  beta={best.beta:.4f}  p={best.pvalue:.4f}  hl={best.half_life:.1f}")

    cfg = PairStrategyConfig(zscore=ZScoreParams(entry=2.0, exit=0.5, stop=4.0, lookback=200))
    strat = PairsStrategy(best, cfg)
    w = strat.generate_weights(closes)

    vec = vectorized_backtest(closes, w, cost=CostModel(fee_bps=4, slippage_bps=2),
                              initial_capital=100_000, timeframe="1h")
    print("\nvectorized backtest:")
    for k, v in vec.metrics.to_dict().items():
        print(f"  {k:<22} {v}")

    evt = event_backtest(closes, strat, cost=CostModel(fee_bps=4, slippage_bps=2),
                        initial_capital=100_000, timeframe="1h", warmup_bars=250)
    print("\nevent-driven backtest:")
    for k, v in evt.metrics.to_dict().items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
