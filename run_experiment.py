"""
run_experiment.py - produce the dispatch tables of the paper from real runs.

Runs the fixed dispatch tool four ways on the same input:

    rule        rule-based engine on realised generation
    milp-causal operational rolling-horizon MILP on the causal forecast
                (seasonal-naive, 24-hour lag; never reads future generation)
    milp        rolling-horizon MILP with perfect foresight (oracle benchmark:
                the optimiser sees the realised series, which no operational
                controller can do - this is an upper bound, not a result)
    milp+fc     rolling-horizon MILP on the forecast-substituted input written
                by forecast.py (perfect-foresight settlement of an external
                forecast; used to value that forecast against the benchmark)

and writes

    <outdir>/table6.tex    annual dispatch outcome, one column per engine
    <outdir>/table8.tex    value of each forecast, normalised against foresight
    <outdir>/kpis.json     the same numbers, machine readable

The value ratio is the only number in the paper that answers "does the forecast
matter", and it is defined as

    value_ratio = (penalty_rule - penalty_forecast)
                / (penalty_rule - penalty_foresight)

so a ratio of 0 means the forecast buys nothing over the heuristic and 1 means
it captures everything the optimiser could capture with perfect information.

Usage
-----
    python run_experiment.py --input synthetic_input.csv --outdir exp_out
    python run_experiment.py --input data.csv --hours 1440 --outdir exp_out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

APP_PATH = Path(__file__).with_name("bess_app_v6.py")


def load_app(path: Path = APP_PATH):
    spec = importlib.util.spec_from_file_location("bess_dispatch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bess_dispatch"] = module
    spec.loader.exec_module(module)
    return module


def build_params(app, args) -> Any:
    return app.BESSParams(
        solar_cap_mw=args.solar_cap,
        wind_cap_mw=args.wind_cap,
        contract_mw=args.contract,
        bess_energy_mwh=args.bess_energy,
        pcs_power_mw=args.bess_power,
        min_soc_pct=args.soc_min,
        max_soc_pct=args.soc_max,
        initial_soc_pct=args.initial_soc,
        peak_hours=[int(h) for h in args.peak_hours.split(",")],
        eta_charge=args.eta_c,
        eta_discharge=args.eta_d,
        grid_limit_mw=args.grid_limit,
        mode="single",
    )


def build_optimizer(app, args, forecast_mode: str) -> Any:
    return app.OptimizerParams(
        horizon_hours=args.horizon,
        step_hours=args.step,
        forecast_mode=forecast_mode,
        forecast_method=args.forecast_method,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        w_penalty=args.w_penalty,
        w_curtail=args.w_curtail,
        w_degradation=args.w_degradation,
        terminal_soc_pct=args.terminal_soc,
        w_terminal_soc=args.w_terminal_soc,
    )


def run_rule(app, df, params) -> Dict[str, float]:
    hourly, _ = app.simulate_single_site(df, params, 1.0)
    return app.kpi_summary(hourly, params)


def run_milp(app, df, params, optimizer) -> Dict[str, float]:
    hourly, _soc, diag = app.simulate_optimized_single_site(
        df, params, optimizer, 1.0, None
    )
    kpis = app.kpi_summary(hourly, params)
    kpis["_windows"] = float(diag.get("windows", 0) or 0)
    kpis["_fallbacks"] = float(diag.get("fallback_windows", 0) or 0)
    kpis["_solve_seconds"] = float(diag.get("solve_seconds", 0.0) or 0.0)
    return kpis


def value_ratio(rule: float, forecast: float, foresight: float,
                min_gain: float = 1.0) -> float:
    """
    Fraction of the optimiser's headroom that the forecast captures.

    The denominator is the gain available with perfect foresight. When that
    gain is negligible the ratio is not a number, it is a division by noise:
    two penalties that both round to zero still differ in the twelfth decimal
    place, and the quotient of two such differences can be any value at all.
    A run in which foresight buys nothing is a legitimate result and it is
    reported as undefined rather than as a spectacular percentage.
    """
    values = (rule, forecast, foresight)
    if not all(np.isfinite(v) for v in values):
        return float("nan")
    denom = rule - foresight
    if denom <= float(min_gain):
        return float("nan")
    return float((rule - forecast) / denom)


def fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:,.{digits}f}"


def write_table6(rows: Dict[str, Dict[str, float]], path: Path) -> None:
    kpis = [
        ("Total generation (MWh)", "Total generation", "MWh", 1),
        ("Total supplied (MWh)", "Total supplied", "MWh", 1),
        ("BESS export (MWh)", "BESS export", "MWh", 1),
        ("Penalty (MWh)", "Penalty", "MWh", 1),
        ("Curtailed (MWh)", "Curtailed", "MWh", 1),
        ("BESS losses (MWh)", "BESS losses", "MWh", 1),
        ("Delivery ratio (%)", "Delivery ratio", "\\%", 2),
    ]
    engines = ("rule", "milp-causal", "milp", "milp+fc")
    lines = [
        "% Auto-generated by run_experiment.py",
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Annual dispatch outcome. ``MILP causal'' is the operational "
        "controller; ``MILP foresight'' is a perfect-information upper bound.}",
        "\\label{tab:dispatch}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}p{2.4cm}cccc@{}}",
        "\\toprule",
        "\\textbf{KPI} & \\textbf{Rule} & \\textbf{MILP causal} & "
        "\\textbf{MILP foresight} & \\textbf{MILP+fc} \\\\",
        "\\midrule",
    ]
    for key, label, unit, digits in kpis:
        cells = [label]
        for engine in engines:
            value = rows.get(engine, {}).get(key)
            cells.append(fmt(value, digits) if value is not None else "--")
        cells.append(unit)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table8(rows: Dict[str, Dict[str, float]], path: Path,
                 min_gain: float = 1.0) -> None:
    rule_pen = rows.get("rule", {}).get("Penalty (MWh)", float("nan"))
    fore_pen = rows.get("milp", {}).get("Penalty (MWh)", float("nan"))
    causal_pen = rows.get("milp-causal", {}).get("Penalty (MWh)", float("nan"))
    fc_pen = rows.get("milp+fc", {}).get("Penalty (MWh)", float("nan"))

    rule_sup = rows.get("rule", {}).get("Total supplied (MWh)", float("nan"))
    fore_sup = rows.get("milp", {}).get("Total supplied (MWh)", float("nan"))
    causal_sup = rows.get("milp-causal", {}).get("Total supplied (MWh)", float("nan"))
    fc_sup = rows.get("milp+fc", {}).get("Total supplied (MWh)", float("nan"))

    ratio_causal = value_ratio(rule_pen, causal_pen, fore_pen, min_gain)
    ratio_fc = value_ratio(rule_pen, fc_pen, fore_pen, min_gain)
    headroom = rule_pen - fore_pen if np.isfinite(rule_pen) and np.isfinite(fore_pen) \
        else float("nan")

    def ratio_cell(value: float) -> str:
        return fmt(value, 3) if np.isfinite(value) else "--"

    caption = ("Value of the forecast. Penalty in MWh; value ratios are normalised "
               "against perfect foresight.")
    note = ""
    if not np.isfinite(ratio_causal) and not np.isfinite(ratio_fc):
        caption += (" The ratio is undefined for this run: perfect foresight does not beat the "
                    f"heuristic by more than {min_gain:g}~MWh, so there is no headroom to "
                    "normalise against. A myopic controller can score worse than the heuristic "
                    "even with perfect information, which is why the headroom is reported "
                    "explicitly.")
        note = ("\n\\par\\smallskip{\\scriptsize Headroom over the heuristic: "
                + fmt(headroom, 2) + "~MWh.}")

    lines = [
        "% Auto-generated by run_experiment.py",
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:value}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "\\textbf{Input} & \\textbf{Penalty} & \\textbf{Supplied} & \\textbf{Value ratio} \\\\",
        "\\midrule",
        f"Rule-based (no forecast) & {fmt(rule_pen)} & {fmt(rule_sup)} & 0.00 \\\\",
        f"Embedded causal forecast & {fmt(causal_pen)} & {fmt(causal_sup)} & {ratio_cell(ratio_causal)} \\\\",
        f"External forecast      & {fmt(fc_pen)} & {fmt(fc_sup)} & {ratio_cell(ratio_fc)} \\\\",
        f"Perfect foresight      & {fmt(fore_pen)} & {fmt(fore_sup)} & 1.00 \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}" + note,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slice_frame(df: pd.DataFrame, start: int, hours: int) -> pd.DataFrame:
    """Apply the same row window to the base and to the forecast inputs."""
    if start:
        df = df.iloc[start:]
    if hours and hours < len(df):
        df = df.iloc[:hours]
    return df.reset_index(drop=True).copy()


def merge_forecast_inputs(
    base: pd.DataFrame, paths: list[Path], start: int
) -> tuple[pd.DataFrame, list[str]]:
    """
    Overlay one or more `fc_input_*.csv` files onto the base window.

    `forecast.py` writes one file per target, each a full copy of the source
    with a single column replaced by its forecast. Taking only the first file
    therefore leaves the other targets at their realised values, and taking
    them by position would misalign the window. Each file is re-sliced with the
    same offset as the base, and only the columns that actually differ are
    copied across. Returns the merged frame and the list of replaced columns.
    """
    merged = base.copy()
    replaced: list[str] = []

    for path in paths:
        frame = pd.read_csv(path) if str(path).lower().endswith(".csv") \
            else pd.read_excel(path)
        frame = slice_frame(frame, start, len(base))
        if len(frame) != len(merged):
            print(f"    WARNING: {path.name} has {len(frame)} rows, "
                  f"base window has {len(merged)}; skipped")
            continue
        for col in merged.columns:
            if col not in frame.columns:
                continue
            a = pd.to_numeric(merged[col], errors="coerce")
            b = pd.to_numeric(frame[col], errors="coerce")
            if a.isna().all() or b.isna().all():
                continue
            if float((a - b).abs().max()) > 1e-9:
                merged[col] = b.to_numpy()
                if col not in replaced:
                    replaced.append(col)

    return merged, replaced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--forecast-input", default="",
                        help="comma-separated fc_input_*.csv files from forecast.py")
    parser.add_argument("--outdir", default="exp_out")
    parser.add_argument("--hours", type=int, default=0, help="0 = whole file")
    parser.add_argument("--start", type=int, default=0,
                        help="row offset of the dispatched window (default 0)")
    parser.add_argument("--solar-cap", type=float, default=281.0)
    parser.add_argument("--wind-cap", type=float, default=0.0)
    parser.add_argument("--contract", type=float, default=190.0)
    parser.add_argument("--bess-energy", type=float, default=600.0)
    parser.add_argument("--bess-power", type=float, default=200.0)
    parser.add_argument("--soc-min", type=float, default=0.0)
    parser.add_argument("--soc-max", type=float, default=90.0)
    parser.add_argument("--initial-soc", type=float, default=50.0,
                        help="initial state of charge, %% of the usable band")
    parser.add_argument("--peak-hours", default="5,6,19,20")
    parser.add_argument("--eta-c", type=float, default=0.97)
    parser.add_argument("--eta-d", type=float, default=0.96)
    parser.add_argument("--grid-limit", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--forecast-method", default="seasonal_naive",
                        choices=("seasonal_naive", "persistence"))
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--w-penalty", type=float, default=1000.0)
    parser.add_argument("--w-curtail", type=float, default=1.0)
    parser.add_argument("--w-degradation", type=float, default=0.0)
    parser.add_argument("--terminal-soc", type=float, default=None,
                        help="hard terminal SoC floor in percent (default: free)")
    parser.add_argument("--w-terminal-soc", type=float, default=0.0,
                        help="soft terminal SoC cost per MWh of deviation")
    parser.add_argument("--min-gain", type=float, default=1.0,
                        help="foresight headroom (MWh) below which the value "
                             "ratio is reported as undefined")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    app = load_app()
    df = pd.read_csv(args.input) if str(args.input).lower().endswith(".csv") \
        else pd.read_excel(args.input)
    df = slice_frame(df, args.start, args.hours)

    params = build_params(app, args)
    optimizer_causal = build_optimizer(app, args, "causal")
    optimizer_oracle = build_optimizer(app, args, "oracle")

    print("=" * 72)
    print("run_experiment.py")
    print(f"  rows        : {len(df)}  (start {args.start})")
    print(f"  engine      : horizon {args.horizon} h / commit {args.step} h")
    print(f"  forecast    : causal ({args.forecast_method}) vs oracle benchmark")
    print(f"  initial SoC : {args.initial_soc:g} % of usable band")
    print(f"  terminal    : "
          f"{'free' if args.terminal_soc is None else str(args.terminal_soc) + ' %'}"
          f"  (soft cost {args.w_terminal_soc:g})")
    print("=" * 72)

    rows: Dict[str, Dict[str, float]] = {}

    started = time.time()
    rows["rule"] = run_rule(app, df, params)
    print(f"  rule        : {time.time() - started:.1f}s  "
          f"penalty {rows['rule']['Penalty (MWh)']:.1f} MWh")

    started = time.time()
    rows["milp-causal"] = run_milp(app, df, params, optimizer_causal)
    print(f"  milp-causal : {time.time() - started:.1f}s  "
          f"penalty {rows['milp-causal']['Penalty (MWh)']:.1f} MWh  "
          f"({rows['milp-causal'].get('_fallbacks', 0):.0f} fallbacks)")

    started = time.time()
    rows["milp"] = run_milp(app, df, params, optimizer_oracle)
    print(f"  milp oracle : {time.time() - started:.1f}s  "
          f"penalty {rows['milp']['Penalty (MWh)']:.1f} MWh")

    if args.forecast_input:
        paths = [Path(p.strip()) for p in args.forecast_input.split(",") if p.strip()]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print("  forecast input not found:", ", ".join(str(p) for p in missing))
            return 2

        fc_df, replaced = merge_forecast_inputs(df, paths, args.start)

        if not replaced:
            print()
            print("  WARNING: the forecast input is identical to the base input over")
            print("           the dispatched window, so milp+fc will equal milp and")
            print("           the value ratio will be meaningless.")
            print("           forecast.py only substitutes the held-out test block;")
            print(f"           re-run with --start near {int(len(pd.read_csv(args.input)) * 0.8)}")
            print("           or dispatch the whole file (--hours 0).")
            print()
        else:
            print(f"  forecast    : replaced {', '.join(replaced)}")

        started = time.time()
        # The external forecast is the information set, so the optimiser is
        # allowed to see it as truth over the window (oracle on fc_df).
        rows["milp+fc"] = run_milp(app, fc_df, params, optimizer_oracle)
        print(f"  milp+fc     : {time.time() - started:.1f}s  "
              f"penalty {rows['milp+fc']['Penalty (MWh)']:.1f} MWh")

    write_table6(rows, outdir / "table6.tex")
    write_table8(rows, outdir / "table8.tex", args.min_gain)
    (outdir / "kpis.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )

    print()
    print("wrote:")
    for name in ("table6.tex", "table8.tex", "kpis.json"):
        print("  ", outdir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
