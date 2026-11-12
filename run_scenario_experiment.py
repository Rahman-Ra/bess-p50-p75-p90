"""
run_scenario_experiment.py
==========================
Run the existing BESS dispatch engines on P50/P75/P90 scenario inputs and
report how the battery KPIs move.

This is the bridge between ``quantile_scenarios.py`` and the dispatch model:
each scenario is a full BESSTOOL input file with ``solar_MWh`` and ``wind_MWh``
replaced by one probability-of-exceedance case.  The physics, constraints and
KPI definitions are exactly those of ``bess_app_v6.py`` -- this script does not
re-implement the battery.

By default the MILP runs in causal mode (seasonal-naive forecast, true receding
horizon). Pass ``--forecast-mode oracle`` to reproduce the perfect-foresight
benchmark instead.

Usage
-----
::

    python run_scenario_experiment.py \
        --inputs scenario_out_cal/battery_input_p50.csv,scenario_out_cal/battery_input_p75.csv,scenario_out_cal/battery_input_p90.csv \
        --labels P50,P75,P90 --engine rule --outdir scenario_dispatch

For the rolling-horizon MILP instead of the rule engine, use ``--engine milp``.
The MILP is slower; start with a single year and ``--hours 8760``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from run_experiment import load_app, slice_frame


def parse_peak_hours(text: str) -> List[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


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
        peak_hours=parse_peak_hours(args.peak_hours),
        eta_charge=args.eta_c,
        eta_discharge=args.eta_d,
        grid_limit_mw=args.grid_limit,
        mode="single",
    )


def build_optimizer(app, args) -> Any:
    return app.OptimizerParams(
        horizon_hours=args.horizon,
        step_hours=args.step,
        forecast_mode=args.forecast_mode,
        forecast_method=args.forecast_method,
        time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        w_penalty=args.w_penalty,
        w_curtail=args.w_curtail,
        w_degradation=args.w_degradation,
        terminal_soc_pct=args.terminal_soc,
        w_terminal_soc=args.w_terminal_soc,
    )


def run_one(app, df: pd.DataFrame, params, optimizer, engine: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if engine == "rule":
        hourly, _ = app.simulate_single_site(df, params, 1.0)
        return app.kpi_summary(hourly, params), {"engine": "rule"}
    hourly, _soc, diag = app.simulate_optimized_single_site(
        df, params, optimizer, 1.0, None
    )
    kpis = app.kpi_summary(hourly, params)
    return kpis, diag


def fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:,.{digits}f}"


def write_outputs(rows: List[Dict[str, Any]], outdir: Path, engine: str) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "scenario_kpis.csv", index=False)
    (outdir / "scenario_kpis.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )

    columns = [
        ("Penalty (MWh)", "Penalty", 1),
        ("Total supplied (MWh)", "Supplied", 1),
        ("BESS export (MWh)", "BESS export", 1),
        ("Curtailed (MWh)", "Curtailed", 1),
        ("BESS losses (MWh)", "BESS losses", 1),
        ("Delivery ratio (%)", "Delivery %", 2),
    ]
    lines = [
        f"# Battery dispatch by generation scenario ({engine})",
        "",
        "| Scenario | " + " | ".join(label for _, label, _ in columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for row in rows:
        cells = [str(row["scenario"])]
        for key, _label, digits in columns:
            cells.append(fmt(row.get(key, float("nan")), digits))
        lines.append("| " + " | ".join(cells) + " |")

    baseline = next((r for r in rows if r.get("is_baseline")), None)
    if baseline is not None:
        lines += ["", f"Baseline: **{baseline['scenario']}**. Deltas below are scenario minus baseline.", ""]
        lines.append("| Scenario | Delta penalty | Delta supplied | Delta BESS export | Delta curtailed |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in rows:
            if row is baseline:
                continue
            lines.append(
                f"| {row['scenario']} | {fmt(row.get('Penalty_delta_vs_baseline', float('nan')))} "
                f"| {fmt(row.get('Supplied_delta_vs_baseline', float('nan')))} "
                f"| {fmt(row.get('BESS_export_delta_vs_baseline', float('nan')))} "
                f"| {fmt(row.get('Curtailed_delta_vs_baseline', float('nan')))} |"
            )
    (outdir / "scenario_kpis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", required=True,
                        help="comma-separated battery input files, one per scenario")
    parser.add_argument("--labels", default="",
                        help="comma-separated labels; default uses file stems")
    parser.add_argument("--engine", default="rule", choices=("rule", "milp"))
    parser.add_argument("--outdir", default="scenario_dispatch")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--hours", type=int, default=0, help="0 = whole file")

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
    parser.add_argument("--forecast-mode", default="causal",
                        choices=("causal", "oracle"),
                        help="causal (operational) or oracle (perfect-foresight benchmark)")
    parser.add_argument("--forecast-method", default="seasonal_naive",
                        choices=("seasonal_naive", "persistence"))
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--w-penalty", type=float, default=1000.0)
    parser.add_argument("--w-curtail", type=float, default=1.0)
    parser.add_argument("--w-degradation", type=float, default=0.0)
    parser.add_argument("--terminal-soc", type=float, default=None)
    parser.add_argument("--w-terminal-soc", type=float, default=0.0)
    parser.add_argument("--baseline-label", default="",
                        help="label used for deltas; default first scenario")
    args = parser.parse_args(argv)

    paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    if not paths:
        print("no input files")
        return 2
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing input files:", ", ".join(str(p) for p in missing))
        return 2

    labels = [v.strip() for v in args.labels.split(",") if v.strip()]
    if not labels:
        labels = [p.stem.replace("battery_input_", "") for p in paths]
    if len(labels) != len(paths):
        print(f"{len(labels)} label(s) for {len(paths)} input file(s)")
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    app = load_app()
    params = build_params(app, args)
    optimizer = build_optimizer(app, args)

    rows: List[Dict[str, Any]] = []
    print("=" * 72)
    print("run_scenario_experiment.py")
    print(f"  engine      : {args.engine}")
    if args.engine == "milp":
        print(f"  forecast    : {args.forecast_mode} ({args.forecast_method})")
        print(f"  horizon     : {args.horizon} h / commit {args.step} h")
        print(f"  initial SoC : {args.initial_soc:g} % of usable band")
    print(f"  scenarios   : {', '.join(labels)}")
    print("=" * 72)

    for label, path in zip(labels, paths):
        frame = pd.read_csv(path) if str(path).lower().endswith(".csv") \
            else pd.read_excel(path)
        frame = slice_frame(frame, args.start, args.hours)
        kpis, diag = run_one(app, frame, params, optimizer, args.engine)
        row: Dict[str, Any] = {
            "scenario": label,
            "file": str(path),
            "engine": args.engine,
            "rows": int(len(frame)),
            **kpis,
        }
        if args.engine == "milp":
            row["windows"] = int(diag.get("windows", 0) or 0)
            row["fallback_windows"] = int(diag.get("fallback_windows", 0) or 0)
            row["forecast_mode"] = args.forecast_mode
            row["forecast_method"] = args.forecast_method
        row["initial_soc_pct"] = args.initial_soc
        rows.append(row)
        print(f"  {label:<12} penalty {kpis.get('Penalty (MWh)', float('nan')):>10,.1f} MWh"
              f"  supplied {kpis.get('Total supplied (MWh)', float('nan')):>12,.1f} MWh")

    baseline_label = args.baseline_label or labels[0]
    baseline = next((r for r in rows if r["scenario"] == baseline_label), rows[0])
    for row in rows:
        row["is_baseline"] = bool(row is baseline)
        row["Penalty_delta_vs_baseline"] = (
            row.get("Penalty (MWh)", float("nan"))
            - baseline.get("Penalty (MWh)", float("nan"))
        )
        row["Supplied_delta_vs_baseline"] = (
            row.get("Total supplied (MWh)", float("nan"))
            - baseline.get("Total supplied (MWh)", float("nan"))
        )
        row["BESS_export_delta_vs_baseline"] = (
            row.get("BESS export (MWh)", float("nan"))
            - baseline.get("BESS export (MWh)", float("nan"))
        )
        row["Curtailed_delta_vs_baseline"] = (
            row.get("Curtailed (MWh)", float("nan"))
            - baseline.get("Curtailed (MWh)", float("nan"))
        )

    write_outputs(rows, outdir, args.engine)
    print("\nwrote:")
    for name in ("scenario_kpis.csv", "scenario_kpis.json", "scenario_kpis.md"):
        print("  ", outdir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
