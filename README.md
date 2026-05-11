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
| `statarb.live` | Production runner — multi-slot PairManager, kill switch, drawdown halt, Telegram alerts |

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
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=0          # 1 = paper trade on real infra with fake money

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
  --exchange binance --timeframe 1h --days 365 --top 50 --pvalue 0.05
```

Outputs a ranked table: `half_life`, `pvalue`, `correlation`, `score`.

### 2 — Backtest every found pair (single window)

```bash
statarb backtest all-pairs \
  --exchange binance --timeframe 1h --days 365 --top 50 \
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
  --exchange binance --timeframe 1h \
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

**Auto-scan mode** (recommended): bot scans the universe on startup, picks the best pairs, and re-scans every `--rescan-days` automatically.

```bash
# 3 parallel pair slots, re-scans every 30 days
statarb paper \
  --exchange binance --timeframe 1h \
  --slots 3 --top 50 \
  --scan-days 90 --rescan-days 30
```

**Fixed-pair mode**: trade a specific pair without re-scanning.

```bash
statarb paper \
  --base ETC/USDT --x CRV/USDT \
  --exchange binance --timeframe 1h
```

### 5 — Live trading

Requires credentials in `.env`. Will ask for confirmation before sending real orders.

```bash
# auto-scan, 3 slots
statarb live \
  --exchange binance --timeframe 1h \
  --slots 3 --top 50 \
  --scan-days 90 --rescan-days 30 \
  --max-drawdown 0.15

# fixed pair
statarb live \
  --base ETC/USDT --x CRV/USDT \
  --exchange binance --timeframe 1h \
  --max-drawdown 0.15
```

Use `--yes` to skip the confirmation prompt (for automated restarts via systemd).

### Stop / restart

```bash
statarb kill     # creates .killswitch — bot stops after current cycle
statarb unkill   # removes kill switch
```

### Verify API credentials

```bash
statarb check-keys --exchange binance
```

---

## Multi-slot position sizing

In auto-scan mode the bot runs N independent pair slots simultaneously.
Each slot contributes `1/N` of total equity to the combined weight vector.

```
equity = 10,000 USDT, slots = 3

slot 0: ETC/CRV   → long  ETC  ~1,500 USDT, short CRV  ~1,800 USDT
slot 1: XRP/BONK  → short XRP  ~1,500 USDT, long  BONK ~1,800 USDT
slot 2: DOGE/AVAX → long  DOGE ~1,500 USDT, short AVAX ~1,800 USDT
```

Within each slot the legs are **dollar-neutral**: long leg ≈ short leg (scaled by the hedge ratio β), so the combined portfolio has minimal directional market exposure.

Slots are staggered so they don't rescan at the same time (slot 0 at bar 0, slot 1 at bar +N×10d, etc.).  When a slot switches pair, the old positions are closed first before the new pair is entered.

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
| Telegram alerts | fill, drawdown warning, pair switch, error, startup/shutdown |
| Exponential backoff | doubles wait time on consecutive errors, caps at 5 min |
| Auto pair re-scan | re-scans universe every N days, closes stale pair, opens best new pair |
| Non-overlapping slots | scanner guarantees no symbol appears in two slots at once |

---

## VPS deployment

### systemd (recommended for production)

```bash
cat > /etc/systemd/system/statarb.service << 'EOF'
[Unit]
Description=statarb live trading bot
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/statarb
ExecStart=/root/statarb/.venv/bin/statarb live \
  --exchange binance --timeframe 1h \
  --slots 3 --top 50 \
  --scan-days 90 --rescan-days 30 \
  --max-drawdown 0.15 --yes
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now statarb
journalctl -u statarb -f          # live logs
```

### tmux (simpler, no auto-restart)

```bash
tmux new -s bot
statarb live --exchange binance --slots 3 --yes
# Ctrl+B then D to detach
tmux attach -t bot                 # reconnect
```

---

## Design notes

- All I/O lives in `data/` and `execution/`. Everything else is pure NumPy/pandas — easy to unit-test without network access.
- The event-driven backtester and the live runner share the same `Strategy` interface. What you backtest is what you trade.
- Scanners are joblib-parallel across CPU cores. Walk-forward runs each fold sequentially (data order matters) but the inner scan is still parallel.
- `panel_closes` requires `min_bars` to prevent newly-listed coins from truncating the full panel via `dropna(how="any")`. Walk-forward sets this automatically from `total_days`.
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
│   ├── live/                  ← runner, PairManager (multi-slot)
│   └── cli.py                 ← Typer CLI entry point
├── tests/                     ← synthetic-data unit tests (no network required)
├── examples/end_to_end.py     ← full programmatic example
├── .env.example               ← credential template
└── pyproject.toml
```

---

## Recommended trading flow

```
scan pairs → walk-forward validate → pick pairs with n_folds ≥ 3 and OOS Sharpe > 2
    → paper trade 2+ weeks (auto-scan mode, same --slots as live)
        → compare fill prices vs backtest assumptions
            → go live (testnet first, then real)
```

Never skip the walk-forward step. In-sample Sharpe is almost always inflated.
