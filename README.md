# statarb

Modular crypto statistical arbitrage toolkit — from universe scanning to live execution.

---

## Modules

| Package | What it does |
|---|---|
| `statarb.data` | CCXT OHLCV fetcher with Parquet cache, resampler, wide-panel builder |
| `statarb.stats` | Engle-Granger, Johansen, ADF, KPSS, Hurst, OLS/TLS/Kalman hedge ratios, OU half-life |
| `statarb.scanner` | Joblib-parallel pairs scanner + Johansen basket scanner |
| `statarb.signals` | Z-score, Bollinger, Kalman-dynamic signal generators |
| `statarb.strategies` | Pairs trading strategy, Johansen-basket strategy |
| `statarb.risk` | Vol-targeting, fractional Kelly, portfolio aggregation, drawdown limits |
| `statarb.backtest` | Vectorized backtester, event-driven engine, walk-forward validator, metrics |
| `statarb.execution` | Paper broker, CCXT live broker (limit orders, fill persistence, reconciliation) |
| `statarb.live` | Production runner (kill switch, drawdown halt, Telegram alerts, backoff) |

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

```ini
# Exchange API keys (required for live trading; data fetch works without them)
BINANCEUS_API_KEY=
BINANCEUS_API_SECRET=
BINANCEUS_TESTNET=0          # 1 = paper trade on real infra with fake money

# Telegram alerts (optional — falls back to console log if unset)
# 1. Message @BotFather → /newbot → copy token
# 2. Message @userinfobot → copy your numeric chat_id
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

STATARB_LOG_LEVEL=INFO
```

---

## Typical workflow

### 1 — Scan for cointegrated pairs

```bash
statarb scan pairs \
  --exchange binanceus --timeframe 1h --days 365 --top 50 --pvalue 0.05
```

Outputs a ranked table: `half_life`, `pvalue`, `correlation`, `score`.

### 2 — Backtest every found pair (single window)

```bash
statarb backtest all-pairs \
  --exchange binanceus --timeframe 1h --days 365 --top 50 \
  --entry 2.0 --exit 0.5 --stop 4.0 --lookback 200 \
  --fee-bps 4 --slippage-bps 2 \
  --sort-by sharpe --out results.csv
```

All returns are **net of fees and slippage**.

### 3 — Walk-forward validation (the honest test)

Scans on a training window, tests on a held-out window, repeats across rolling folds.
A pair with consistently positive OOS Sharpe is likely a real edge; in-sample flukes disappear.

```bash
statarb backtest walk-forward \
  --exchange binanceus --timeframe 1h \
  --total-days 365 --scan-days 90 --test-days 30 \
  --top 50 --entry 2.0 --exit 0.5 --stop 4.0 --lookback 200 \
  --fee-bps 4 --slippage-bps 2 \
  --min-folds 3 --out wf_results.csv
```

Key columns in the output:

| column | meaning |
|---|---|
| `n_folds` | how many folds the pair was found and tested |
| `mean_oos_sharpe` | average OOS Sharpe — the number that matters |
| `pct_folds_positive` | fraction of folds with positive return — want > 0.6 |
| `worst_oos_dd` | worst OOS drawdown across folds |

### 4 — Paper trade (dry run)

Runs the full live loop — generates weights, logs orders — but never touches the exchange.

```bash
statarb paper \
  --base SOL/USDT --x LTC/USDT \
  --exchange binanceus --timeframe 1h
```

### 5 — Live trading

Requires credentials in `.env`. Will ask for confirmation before sending real orders.

```bash
statarb live \
  --base SOL/USDT --x LTC/USDT \
  --exchange binanceus --timeframe 1h \
  --max-drawdown 0.15
```

Use `--yes` to skip the confirmation prompt (for automated restarts).

### Stop / restart

```bash
statarb kill     # creates .killswitch — bot stops after current cycle
statarb unkill   # removes kill switch
```

### Verify API credentials

```bash
statarb check-keys --exchange binanceus
```

---

## Backtest metrics reference

| metric | unit | healthy range |
|---|---|---|
| `total_return` | fraction | — |
| `cagr` | fraction / yr | > 0.15 |
| `annual_vol` | fraction / yr | — |
| `sharpe` | ratio | > 1.0 |
| `sortino` | ratio | > 1.0 |
| `calmar` | ratio | > 1.0 |
| `max_drawdown` | fraction | > −0.20 |
| `turnover_annual` | × NAV / yr | < 300 |
| `n_trades` | count | > 30 |

**turnover drag** = `turnover_annual × (fee_bps + slippage_bps) / 10_000`
e.g. 300 × 6 bps = **18% annual cost** — subtract from gross return to get net.

---

## Live execution safety features

| feature | detail |
|---|---|
| Limit orders + market fallback | posts aggressive limit, cancels after 45s, retries market |
| Fill persistence | every fill written to SQLite (`fills.db`) — survives restarts |
| Position reconciliation | on startup, syncs in-memory state against actual exchange balance |
| Drawdown halt | stops trading if equity drops > `max_drawdown` from peak |
| Kill switch | `statarb kill` → graceful stop at end of current bar |
| Telegram alerts | fill notifications, drawdown warnings, error alerts, startup/shutdown |
| Exponential backoff | doubles wait time on consecutive errors, caps at 5 min |

---

## Design notes

- All I/O lives in `data/` and `execution/`. Everything else is pure NumPy/pandas — easy to unit-test without network access.
- The event-driven backtester and the live runner share the same `Strategy` interface. What you backtest is what you trade.
- Scanners are joblib-parallel across CPU cores. Walk-forward runs each fold sequentially (data order matters) but the inner scan is still parallel.
- Credentials never appear in logs. The broker refuses `dry_run=False` without a key.

---

## Project layout

```
statarb/
├── config/default.yaml        ← default params (override via .env or CLI)
├── statarb/
│   ├── core/                  ← types, config loader, secrets, logging
│   ├── data/                  ← CCXT provider, Parquet cache, resampler
│   ├── universe/              ← liquidity / history filters
│   ├── stats/                 ← cointegration, stationarity, hedge ratios, OU
│   ├── scanner/               ← pairs scanner, basket scanner
│   ├── signals/               ← z-score, Bollinger, Kalman signal generators
│   ├── strategies/            ← pairs strategy, basket strategy
│   ├── risk/                  ← sizing, portfolio aggregation, limits
│   ├── backtest/              ← vectorized, event-driven, walk-forward, metrics
│   ├── execution/             ← paper broker, CCXT live broker, alerts, persistence
│   ├── live/                  ← production runner
│   └── cli.py                 ← Typer CLI entry point
├── tests/                     ← synthetic-data unit tests (no network required)
├── examples/end_to_end.py     ← full programmatic example
├── .env.example               ← credential template
└── pyproject.toml
```

---

## Recommended trading flow

```
fetch data → scan pairs → walk-forward validate
    → paper trade 2+ weeks → compare execution vs backtest
        → go live (testnet first, then real)
```

Never skip the walk-forward step. In-sample Sharpe is almost always inflated.
