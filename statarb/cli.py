"""statarb CLI.

Subcommands:
- fetch   : warm the OHLCV cache for a top-N universe
- scan    : pair or basket scanner over the cached universe
- backtest: vectorized backtest of a single pair
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .backtest.costs import CostModel
from .backtest.vectorized import vectorized_backtest
from .core.config import load_config
from .core.logging import get_logger
from .core.secrets import get_credentials
from .core.types import PairSpec
from .data.ccxt_provider import CCXTProvider, panel_closes
from .execution.alerts import Alerter
from .execution.ccxt_exec import CCXTBroker
from .live.runner import KILLSWITCH, RunnerConfig, run_live
from .scanner.pairs_scanner import PairsScanConfig, pairs_to_dataframe, scan_pairs
from .strategies.pairs_trading import PairsStrategy, PairStrategyConfig
from .universe.filters import filter_universe

app = typer.Typer(add_completion=False, help="statarb — modular crypto stat-arb toolkit")
scan_app = typer.Typer(help="Scanners")
bt_app = typer.Typer(help="Backtesters")
app.add_typer(scan_app, name="scan")
app.add_typer(bt_app, name="backtest")
console = Console()
log = get_logger("cli")


@app.command("check-keys")
def check_keys(exchange: str = "binance"):
    """Verify .env credentials by hitting an authenticated endpoint."""
    creds = get_credentials(exchange)
    if not creds.has_credentials:
        console.print(f"[red]no credentials for {exchange}[/red] — set "
                      f"{exchange.upper()}_API_KEY / _API_SECRET in .env")
        raise typer.Exit(1)
    provider = CCXTProvider(exchange=exchange)
    try:
        bal = provider.ex.fetch_balance()
    except Exception as e:
        console.print(f"[red]auth call failed:[/red] {e}")
        raise typer.Exit(2)
    non_zero = {k: v for k, v in bal.get("total", {}).items() if v}
    t = Table(title=f"{exchange} balances (testnet={creds.testnet})")
    t.add_column("asset"); t.add_column("total")
    for k, v in sorted(non_zero.items()):
        t.add_row(k, f"{v:.6f}")
    if not non_zero:
        t.add_row("(empty)", "0")
    console.print(t)


@app.command()
def fetch(
    exchange: str = "binance",
    quote: str = "USDT",
    top: int = 50,
    timeframe: str = "1h",
    days: int = 365,
):
    """Pre-warm the local OHLCV cache for the top-N symbols by volume."""
    provider = CCXTProvider(exchange=exchange)
    symbols = provider.top_by_volume(n=top, quote=quote)
    console.print(f"top {top} by {quote} volume on {exchange}: {symbols[:5]} ...")
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    for s in symbols:
        df = provider.fetch_ohlcv(s, timeframe, start=start, end=end)
        console.print(f"{s:<14} {len(df):>6} bars")


@scan_app.command("pairs")
def scan_pairs_cmd(
    exchange: str = "binance",
    quote: str = "USDT",
    timeframe: str = "1h",
    days: int = 365,
    top: int = 50,
    pvalue: float = 0.05,
    workers: int = 8,
    min_history: int | None = None,
    out: Path | None = None,
):
    """Scan the top-N symbols for cointegrated pairs."""
    cfg = load_config()
    provider = CCXTProvider(exchange=exchange)
    console.print(f"[dim]testnet={provider.credentials.testnet} auth={provider.credentials.has_credentials}[/dim]")
    symbols = provider.top_by_volume(n=top, quote=quote)
    console.print(f"top_by_volume → {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    if not symbols:
        console.print("[red]no symbols returned. likely causes: geo-block, "
                      "testnet=1 (limited markets), or wrong --quote.[/red]")
        raise typer.Exit(2)
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    mh = min_history if min_history is not None else cfg["universe"]["min_history_bars"]
    closes = panel_closes(provider, symbols, timeframe, start=start, end=end, min_bars=mh)
    console.print(f"panel_closes → {closes.shape[1]} columns × {len(closes)} rows")
    if closes.empty:
        console.print("[red]no OHLCV data; check exchange / timeframe.[/red]")
        raise typer.Exit(3)
    universe = filter_universe(closes, min_history_bars=mh, exclude=cfg["universe"]["exclude_symbols"])
    console.print(f"filter_universe(min_history={mh}) → {len(universe)} symbols")
    closes = closes[universe]
    if closes.shape[1] < 2:
        console.print("[red]not enough symbols survived filtering — lower --min-history "
                      "or --days to widen the window.[/red]")
        raise typer.Exit(4)
    pairs = scan_pairs(closes, PairsScanConfig(pvalue_max=pvalue, workers=workers))
    df = pairs_to_dataframe(pairs)
    _print_pairs_table(df.head(25))
    if out:
        df.to_csv(out, index=False)
        console.print(f"wrote {out}")


@bt_app.command("pair")
def backtest_pair(
    base: str = typer.Option(..., "--base", help="y-leg symbol, e.g. ETH/USDT"),
    x: str = typer.Option(..., "--x", help="x-leg symbol, e.g. BTC/USDT"),
    exchange: str = "binance",
    timeframe: str = "1h",
    days: int = 365,
    capital: float = 100_000.0,
    entry: float = 2.0,
    exit_: float = typer.Option(0.5, "--exit"),
    stop: float = 4.0,
    lookback: int = 200,
    use_kalman: bool = False,
    fee_bps: float = typer.Option(4.0, help="taker fee per side, basis points"),
    slippage_bps: float = typer.Option(2.0, help="slippage per side, basis points"),
):
    """Backtest a single pairs-trading strategy."""
    provider = CCXTProvider(exchange=exchange)
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    closes = panel_closes(provider, [base, x], timeframe, start=start, end=end)
    if closes.empty:
        console.print("[red]no data; check symbols[/red]")
        raise typer.Exit(1)

    from .stats.cointegration import engle_granger
    import numpy as np
    eg = engle_granger(np.log(closes[base]), np.log(closes[x]))
    pair = PairSpec(y=base, x=x, beta=eg.beta, alpha=eg.alpha, half_life=eg.half_life, pvalue=eg.pvalue)
    console.print(f"pair fit: beta={eg.beta:.4f}, p={eg.pvalue:.4f}, half-life={eg.half_life:.1f} bars")

    from .signals.zscore import ZScoreParams
    cfg = PairStrategyConfig(
        use_kalman=use_kalman,
        zscore=ZScoreParams(entry=entry, exit=exit_, stop=stop, lookback=lookback),
    )
    strat = PairsStrategy(pair, cfg)
    weights = strat.generate_weights(closes)
    cost = CostModel(fee_bps=fee_bps, slippage_bps=slippage_bps)
    res = vectorized_backtest(closes, weights, cost=cost, initial_capital=capital, timeframe=timeframe)
    console.print(f"[dim]costs: fee={fee_bps}bps slippage={slippage_bps}bps "
                  f"(~{(fee_bps + slippage_bps) * res.metrics.turnover_annual / 10_000:.1%} drag/yr)[/dim]")
    _print_metrics_table(res.metrics.to_dict())


def _build_pair_strategy(provider, base, x, timeframe, history_days, use_kalman):
    import numpy as np
    from .signals.zscore import ZScoreParams
    from .stats.cointegration import engle_granger
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=history_days)
    closes = panel_closes(provider, [base, x], timeframe, start=start, end=end)
    if closes.empty:
        console.print("[red]no data — check symbols / exchange[/red]")
        raise typer.Exit(1)
    eg = engle_granger(np.log(closes[base]), np.log(closes[x]))
    console.print(f"pair fit: beta={eg.beta:.4f}  p={eg.pvalue:.4f}  hl={eg.half_life:.1f} bars")
    pair = PairSpec(y=base, x=x, beta=eg.beta, alpha=eg.alpha, half_life=eg.half_life, pvalue=eg.pvalue)
    return PairsStrategy(pair, PairStrategyConfig(use_kalman=use_kalman, zscore=ZScoreParams()))


def _make_strategy_factory(timeframe, history_days, use_kalman, entry, exit_, stop, lookback):
    """Return a callable (PairSpec) -> Strategy for the auto-scan runner."""
    from .signals.zscore import ZScoreParams
    cfg = PairStrategyConfig(
        use_kalman=use_kalman,
        zscore=ZScoreParams(entry=entry, exit=exit_, stop=stop, lookback=lookback),
    )
    def factory(pair: PairSpec):
        return PairsStrategy(pair, cfg)
    return factory


@app.command()
def paper(
    base: str = typer.Option(None, "--base", help="fixed base symbol (omit for auto-scan)"),
    x: str = typer.Option(None, "--x", help="fixed quote symbol (omit for auto-scan)"),
    exchange: str = "binance",
    quote: str = "USDT",
    timeframe: str = "1h",
    history_days: int = 60,
    use_kalman: bool = False,
    max_drawdown: float = 0.20,
    # auto-scan options
    auto_scan: bool = typer.Option(True, "--auto-scan/--no-auto-scan",
                                   help="periodically re-scan universe and switch pair"),
    top: int = typer.Option(50, help="universe size for auto-scan"),
    scan_days: int = typer.Option(90, help="training window per scan (days)"),
    rescan_days: int = typer.Option(30, help="how often to re-scan (days)"),
    pvalue: float = 0.05,
    entry: float = 2.0,
    exit_: float = typer.Option(0.5, "--exit"),
    stop: float = 4.0,
    lookback: int = 200,
):
    """Dry-run loop — computes weights and logs orders but never touches exchange."""
    from .live.pair_manager import PairManager, PairManagerConfig

    provider = CCXTProvider(exchange=exchange)
    alerter = Alerter()
    broker = CCXTBroker(exchange=exchange, dry_run=True, alerter=alerter)

    if base and x and not auto_scan:
        # fixed-pair mode
        strat = _build_pair_strategy(provider, base, x, timeframe, history_days, use_kalman)
        cfg = RunnerConfig(symbols=[base, x], timeframe=timeframe, max_drawdown=max_drawdown)
        run_live(provider, broker, strat, cfg, alerter)
    else:
        # auto-scan mode
        pm_cfg = PairManagerConfig(
            universe_top=top, quote=quote,
            scan_days=scan_days, rescan_interval_days=rescan_days,
            pvalue_max=pvalue,
        )
        pm = PairManager(provider, pm_cfg, timeframe)
        factory = _make_strategy_factory(timeframe, history_days, use_kalman, entry, exit_, stop, lookback)
        cfg = RunnerConfig(symbols=[], timeframe=timeframe, max_drawdown=max_drawdown)
        console.print(f"[green]auto-scan mode[/green]: top-{top}, scan={scan_days}d, rescan every {rescan_days}d")
        run_live(provider, broker, None, cfg, alerter, pair_manager=pm, strategy_factory=factory)


@app.command()
def live(
    base: str = typer.Option(None, "--base", help="fixed base symbol (omit for auto-scan)"),
    x: str = typer.Option(None, "--x", help="fixed quote symbol (omit for auto-scan)"),
    exchange: str = "binance",
    quote: str = "USDT",
    timeframe: str = "1h",
    history_days: int = 60,
    use_kalman: bool = False,
    max_drawdown: float = 0.15,
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation prompt"),
    # auto-scan options
    auto_scan: bool = typer.Option(True, "--auto-scan/--no-auto-scan",
                                   help="periodically re-scan universe and switch pair"),
    top: int = typer.Option(50, help="universe size for auto-scan"),
    scan_days: int = typer.Option(90, help="training window per scan (days)"),
    rescan_days: int = typer.Option(30, help="how often to re-scan (days)"),
    pvalue: float = 0.05,
    entry: float = 2.0,
    exit_: float = typer.Option(0.5, "--exit"),
    stop: float = 4.0,
    lookback: int = 200,
):
    """Live trading with real orders. Requires credentials in .env."""
    from .live.pair_manager import PairManager, PairManagerConfig

    creds = get_credentials(exchange)
    if not creds.has_credentials:
        console.print(f"[red]no credentials for {exchange} — add to .env first[/red]")
        raise typer.Exit(1)

    fixed_mode = bool(base and x and not auto_scan)
    pair_label = f"{base} / {x}" if fixed_mode else f"auto-scan top-{top}"

    if not yes:
        console.print(
            f"\n[bold yellow]⚠  LIVE TRADING[/bold yellow]\n"
            f"exchange : [bold]{exchange}[/bold]  testnet={creds.testnet}\n"
            f"pair     : [bold]{pair_label}[/bold]\n"
            f"halt at  : drawdown > [bold]{max_drawdown:.0%}[/bold]\n"
        )
        confirm = typer.confirm("Send real orders?", default=False)
        if not confirm:
            raise typer.Abort()

    provider = CCXTProvider(exchange=exchange)
    alerter = Alerter()
    broker = CCXTBroker(exchange=exchange, dry_run=False, alerter=alerter)

    if fixed_mode:
        strat = _build_pair_strategy(provider, base, x, timeframe, history_days, use_kalman)
        cfg = RunnerConfig(symbols=[base, x], timeframe=timeframe, max_drawdown=max_drawdown)
        run_live(provider, broker, strat, cfg, alerter)
    else:
        pm_cfg = PairManagerConfig(
            universe_top=top, quote=quote,
            scan_days=scan_days, rescan_interval_days=rescan_days,
            pvalue_max=pvalue,
        )
        pm = PairManager(provider, pm_cfg, timeframe)
        factory = _make_strategy_factory(timeframe, history_days, use_kalman, entry, exit_, stop, lookback)
        cfg = RunnerConfig(symbols=[], timeframe=timeframe, max_drawdown=max_drawdown)
        run_live(provider, broker, None, cfg, alerter, pair_manager=pm, strategy_factory=factory)


@app.command()
def kill():
    """Create .killswitch — running bot will stop after the current cycle."""
    KILLSWITCH.touch()
    console.print(f"[yellow]kill switch set ({KILLSWITCH.resolve()})[/yellow]\n"
                  "bot will stop at end of current cycle.")


@app.command()
def unkill():
    """Remove .killswitch so the bot can run again."""
    if KILLSWITCH.exists():
        KILLSWITCH.unlink()
        console.print("[green]kill switch removed — bot will resume on next start[/green]")
    else:
        console.print("no kill switch found")


@bt_app.command("all-pairs")
def backtest_all_pairs(
    exchange: str = "binance",
    quote: str = "USDT",
    timeframe: str = "1h",
    days: int = 365,
    top: int = 50,
    pvalue: float = 0.05,
    entry: float = 2.0,
    exit_: float = typer.Option(0.5, "--exit"),
    stop: float = 4.0,
    lookback: int = 200,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    min_history: int = 2000,
    sort_by: str = typer.Option("sharpe", help="sharpe | calmar | total_return | max_drawdown"),
    top_n: int = typer.Option(20, help="how many pairs to show in ranking table"),
    workers: int = 4,
    out: Path | None = None,
):
    """Scan + backtest every cointegrated pair, then rank by performance metric."""
    from joblib import Parallel, delayed

    # ---- 1. fetch panel ----
    cfg_ = load_config()
    provider = CCXTProvider(exchange=exchange)
    symbols = provider.top_by_volume(n=top, quote=quote)
    console.print(f"fetching {len(symbols)} symbols …")
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    closes = panel_closes(provider, symbols, timeframe, start=start, end=end, min_bars=min_history)
    console.print(f"panel: {closes.shape[1]} × {len(closes)} bars")
    universe = filter_universe(closes, min_history_bars=min_history,
                               exclude=cfg_["universe"]["exclude_symbols"])
    closes = closes[universe]
    console.print(f"universe after filter: {len(universe)} symbols")
    if len(universe) < 2:
        console.print("[red]not enough symbols — lower --min-history or --days[/red]")
        raise typer.Exit(1)

    # ---- 2. scan ----
    pairs = scan_pairs(closes, PairsScanConfig(pvalue_max=pvalue, workers=workers))
    if not pairs:
        console.print("[red]no cointegrated pairs found[/red]")
        raise typer.Exit(2)
    console.print(f"found [bold]{len(pairs)}[/bold] pairs — backtesting each …")

    # ---- 3. backtest each pair ----
    cost = CostModel(fee_bps=fee_bps, slippage_bps=slippage_bps)
    from .signals.zscore import ZScoreParams

    def _bt_one(pair: PairSpec) -> dict:
        try:
            sub = closes[[pair.y, pair.x]].dropna()
            if len(sub) < lookback + 50:
                return {}
            cfg_p = PairStrategyConfig(zscore=ZScoreParams(
                entry=entry, exit=exit_, stop=stop, lookback=lookback))
            w = PairsStrategy(pair, cfg_p).generate_weights(sub)
            res = vectorized_backtest(sub, w, cost=cost, timeframe=timeframe)
            m = res.metrics.to_dict()
            m.update({"y": pair.y, "x": pair.x,
                       "half_life": round(pair.half_life, 1),
                       "pvalue": round(pair.pvalue, 4),
                       "correlation": round(pair.meta.get("correlation", 0), 3)})
            return m
        except Exception:
            return {}

    rows = Parallel(n_jobs=workers, prefer="threads")(
        delayed(_bt_one)(p) for p in pairs
    )
    rows = [r for r in rows if r]
    if not rows:
        console.print("[red]all backtests failed[/red]")
        raise typer.Exit(3)

    # ---- 4. rank + display ----
    df = pd.DataFrame(rows)
    asc = sort_by == "max_drawdown"
    df = df.sort_values(sort_by, ascending=asc).head(top_n)

    display_cols = ["y", "x", "half_life", "pvalue", "correlation",
                    "sharpe", "total_return", "max_drawdown", "turnover_annual", "n_trades"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_show = df[display_cols].reset_index(drop=True)

    t = Table(title=f"Pair backtest ranking (sorted by {sort_by})")
    for c in df_show.columns:
        t.add_column(c, justify="right" if c not in ("y", "x") else "left")
    for _, row in df_show.iterrows():
        t.add_row(*[
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ])
    console.print(t)
    console.print(f"[dim]costs: {fee_bps}bps fee + {slippage_bps}bps slippage | "
                  f"{len(rows)} pairs backtested[/dim]")

    if out:
        df.to_csv(out, index=False)
        console.print(f"saved → {out}")


@bt_app.command("walk-forward")
def backtest_walk_forward(
    exchange: str = "binance",
    quote: str = "USDT",
    timeframe: str = "1h",
    total_days: int = typer.Option(180, help="total history to fetch"),
    scan_days: int = typer.Option(60, help="training window per fold (days)"),
    test_days: int = typer.Option(20, help="OOS test window per fold (days)"),
    step_days: int = typer.Option(0, help="roll step between folds — 0 = same as test_days"),
    top: int = 50,
    pvalue: float = 0.05,
    entry: float = 2.0,
    exit_: float = typer.Option(0.5, "--exit"),
    stop: float = 4.0,
    lookback: int = 200,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    min_history: int = 500,
    workers: int = 4,
    min_folds: int = typer.Option(2, help="hide pairs that appear in fewer folds than this"),
    out: Path | None = None,
):
    """Scan + OOS backtest across rolling folds — the honest test."""
    from .backtest.walkforward import run_walk_forward
    from .signals.zscore import ZScoreParams
    from .strategies.pairs_trading import PairStrategyConfig

    cfg_ = load_config()

    tf_bars = {
        "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
        "1h": 24, "4h": 6, "1d": 1,
    }
    bars_per_day = tf_bars.get(timeframe, 24)
    scan_bars  = scan_days  * bars_per_day
    test_bars  = test_days  * bars_per_day
    step_bars_ = (step_days or test_days) * bars_per_day

    # require each symbol to cover ≥85 % of the full requested history so
    # recently-listed coins don't truncate the whole panel via dropna(how="any")
    min_bars_wf = int(total_days * bars_per_day * 0.85)

    provider = CCXTProvider(exchange=exchange)
    symbols = provider.top_by_volume(n=top, quote=quote)
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=total_days)
    console.print(f"fetching {len(symbols)} symbols × {total_days}d …")
    closes = panel_closes(provider, symbols, timeframe, start=start, end=end, min_bars=min_bars_wf)
    console.print(f"panel: {closes.shape[1]} × {len(closes)} bars")

    if len(closes) < scan_bars + test_bars:
        needed = scan_bars + test_bars
        console.print(
            f"[red]panel too short for even 1 fold: "
            f"{len(closes)} bars available, need {needed} "
            f"(scan {scan_bars} + test {test_bars}).[/red]\n"
            f"[yellow]Try: --total-days {(needed // bars_per_day) + 10} "
            f"or --scan-days {scan_days // 2} --test-days {test_days // 2}[/yellow]"
        )
        raise typer.Exit(1)

    universe = filter_universe(closes, min_history_bars=min_bars_wf,
                               exclude=cfg_["universe"]["exclude_symbols"])
    closes = closes[universe]
    console.print(f"universe: {len(universe)} symbols")

    n_folds_expected = max(0, (len(closes) - scan_bars) // step_bars_)
    console.print(
        f"walk-forward: scan={scan_days}d ({scan_bars} bars)  "
        f"test={test_days}d ({test_bars} bars)  "
        f"step={step_days or test_days}d  ~{n_folds_expected} folds"
    )

    scan_cfg = PairsScanConfig(pvalue_max=pvalue, workers=workers, show_progress=False)
    strat_cfg = PairStrategyConfig(
        zscore=ZScoreParams(entry=entry, exit=exit_, stop=stop, lookback=lookback)
    )
    cost = CostModel(fee_bps=fee_bps, slippage_bps=slippage_bps)

    wf = run_walk_forward(closes, scan_cfg, strat_cfg, cost, scan_bars, test_bars, step_bars_, timeframe)

    if wf.pair_summary.empty:
        console.print("[red]no OOS results — try wider windows or more data[/red]")
        raise typer.Exit(1)

    # ---- fold summary ----
    t1 = Table(title="Per-fold OOS summary")
    for c in wf.fold_summary.columns:
        t1.add_column(c)
    for _, row in wf.fold_summary.iterrows():
        t1.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t1)

    # ---- pair summary (filter by min_folds) ----
    ps = wf.pair_summary[wf.pair_summary["n_folds"] >= min_folds]
    t2 = Table(title=f"Pair OOS ranking (≥{min_folds} folds, sorted by mean OOS Sharpe)")
    for c in ps.columns:
        t2.add_column(c, justify="right" if c not in ("y", "x") else "left")
    for _, row in ps.iterrows():
        sharpe_color = (
            "green" if row.get("mean_oos_sharpe", 0) > 1
            else "yellow" if row.get("mean_oos_sharpe", 0) > 0
            else "red"
        )
        vals = []
        for k, v in row.items():
            if isinstance(v, float):
                fmt = f"{v:.4f}"
            else:
                fmt = str(v)
            vals.append(f"[{sharpe_color}]{fmt}[/{sharpe_color}]" if k == "mean_oos_sharpe" else fmt)
        t2.add_row(*vals)
    console.print(t2)
    console.print(
        f"[dim]{len(wf.folds)} folds completed | "
        f"costs: {fee_bps}bps fee + {slippage_bps}bps slippage[/dim]"
    )

    if out:
        wf.pair_summary.to_csv(out, index=False)
        console.print(f"saved → {out}")


def _print_pairs_table(df: pd.DataFrame) -> None:
    t = Table(title="Top cointegrated pairs")
    for c in df.columns:
        t.add_column(c)
    for _, row in df.iterrows():
        t.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


def _print_metrics_table(m: dict) -> None:
    t = Table(title="Backtest metrics")
    t.add_column("metric")
    t.add_column("value")
    for k, v in m.items():
        t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(t)


if __name__ == "__main__":
    app()
