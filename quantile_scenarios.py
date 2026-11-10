"""
quantile_scenarios.py
=====================
Turn deterministic P50 wind/solar forecasts into P75 and P90
probability-of-exceedance scenarios, and write battery-ready inputs.

Convention
----------
P90 is the *exceedance* value: generation is expected to be at least this
value in 90 % of cases.  It is the 10th percentile of the predictive
distribution, so for generation::

    P90 <= P75 <= P50

If you want the optimistic "upside" value instead, use ``--side upper`` and
the labels become P10/P25/P50.  Do not mix the two conventions in one run.

How the conversion works
------------------------
The P50 forecast is the central forecast.  Forecast errors are formed on a
calibration window::

    r(t) = actual(t) - P50(t)

Then, for an exceedance level L (50, 75, 90)::

    P_L(t) = P50(t) + q_(1-L/100)( r | month, hour )

The residual quantile is estimated globally and inside month/hour groups; a
small group is shrunk towards the global quantile so that a group with a
handful of hours cannot set a wild offset.  This is deliberately simple and
auditable: every offset has a direct interpretation in MWh.

A multiplicative mode is available for targets that never go to zero, but
hourly solar has many zero/near-zero hours; a ratio quantile then collapses
to zero and is useless.  Additive calibration is therefore the default.

Why not a fixed haircut?
------------------------
A fixed 0.8 x P50 is not a probability statement.  It over-penalises stable
hours and under-protects uncertain ones.  On the out-of-sample predictions in
this repository, a fixed 0.8 x P50 gave roughly 45 % coverage, while the
empirical P90 gave 89-90 %.

Joint solar+wind
----------------
For a shared battery, do not add solar-P90 and wind-P90: that double-counts
uncertainty when the two resources are imperfectly correlated.  ``--aggregation
joint`` calibrates the *total* solar+wind generation and allocates the total
scenario back to the two technologies in proportion to their P50 values.  Use
``--aggregation individual`` only when each technology has its own battery.

Typical use
-----------
::

    python quantile_scenarios.py \
        --solar-pred oos_out/lgbm_solar_pred.csv \
        --wind-pred  oos_out/lgbm_wind_pred.csv \
        --train-years 2020,2021,2022 --test-years 2023,2024 \
        --group month,hour --aggregation joint \
        --base-input real_input.csv --apply-year 2024 \
        --outdir scenario_out

Outputs
-------
``scenario_hourly.csv``
    time, actual/P50/P75/P90 for solar, wind and the combined plant.
``scenario_summary.csv`` / ``.json``
    coverage, bias, pinball loss and mean values for each scenario.
``scenario_calibration.json``
    the fitted offsets, group counts and run settings (reproducibility).
``battery_input_p50.csv``, ``battery_input_p75.csv``, ``battery_input_p90.csv``
    full BESSTOOL input files with ``solar_MWh`` and ``wind_MWh`` replaced by
    the selected scenario over the requested year.  The dispatch app can read
    these directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_LEVELS: Tuple[int, ...] = (50, 75, 90)
VALID_GROUPS = ("month", "hour", "dayofyear", "weekday")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_float(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                      dtype=float)


def _quantile(values: np.ndarray, q: float) -> float:
    """NaN-safe quantile; an empty sample calibrates to zero."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _time_columns(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy with a parsed ``time`` column and calendar fields."""
    out = frame.copy()
    if time_col in out.columns:
        out["time"] = pd.to_datetime(out[time_col], errors="coerce")
    elif {"Year", "Month", "Day", "Hour"}.issubset(out.columns):
        out["time"] = pd.to_datetime(
            {
                "year": pd.to_numeric(out["Year"], errors="coerce"),
                "month": pd.to_numeric(out["Month"], errors="coerce"),
                "day": pd.to_numeric(out["Day"], errors="coerce"),
                "hour": pd.to_numeric(out["Hour"], errors="coerce"),
            },
            errors="coerce",
        )
    else:
        raise ValueError(
            f"no '{time_col}' column and no Year/Month/Day/Hour columns"
        )
    if out["time"].isna().any():
        bad = int(out["time"].isna().sum())
        raise ValueError(f"{bad} row(s) have an invalid timestamp")
    out["month"] = out["time"].dt.month
    out["hour"] = out["time"].dt.hour
    out["dayofyear"] = out["time"].dt.dayofyear
    out["weekday"] = out["time"].dt.weekday
    return out


def load_prediction_file(
    path: Path,
    *,
    time_col: str = "time",
    actual_col: str = "actual",
    p50_col: str = "pred",
) -> pd.DataFrame:
    """Load one forecast file and normalise it to time/actual/p50."""
    frame = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    frame = _time_columns(frame, time_col)
    if actual_col not in frame.columns:
        raise ValueError(f"{path} has no actual column '{actual_col}'")
    if p50_col not in frame.columns:
        raise ValueError(f"{path} has no P50 column '{p50_col}'")
    out = frame[["time", "month", "hour", "dayofyear", "weekday",
                 actual_col, p50_col]].copy()
    out = out.rename(columns={actual_col: "actual", p50_col: "p50"})
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["p50"] = pd.to_numeric(out["p50"], errors="coerce")
    out = out.dropna(subset=["time", "actual", "p50"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def _group_key_frame(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    if not group_cols:
        return pd.Series([()] * len(frame), index=frame.index, dtype=object)
    if len(group_cols) == 1:
        return frame[group_cols[0]].apply(lambda v: (int(v),))
    return frame[list(group_cols)].apply(
        lambda row: tuple(int(v) for v in row), axis=1
    )


# ---------------------------------------------------------------------------
# calibration object
# ---------------------------------------------------------------------------
@dataclass
class ExceedanceCalibration:
    """Fitted residual quantiles for one target."""

    target: str
    levels: Tuple[int, ...]
    method: str                      # "additive" or "multiplicative"
    group_cols: Tuple[str, ...]
    global_values: Dict[int, float]
    group_values: Dict[Tuple[int, ...], Dict[int, float]] = field(default_factory=dict)
    group_counts: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    min_samples: int = 200
    shrinkage: float = 50.0
    zero_floor: float = 1e-6
    calibrate_p50: bool = False

    def _value_for_key(self, level: int, key: Tuple[int, ...]) -> float:
        group = self.group_values.get(key)
        if group is not None and level in group:
            return float(group[level])
        return float(self.global_values.get(level, 0.0))

    def value(self, level: int, *, month=None, hour=None,
              dayofyear=None, weekday=None) -> Any:
        """
        Return the calibration value for a level.  Scalars and arrays are both
        accepted; the return has the same shape as the supplied calendar data.
        """
        if level not in self.levels:
            raise KeyError(f"level P{level} was not fitted; fitted {self.levels}")
        if not self.group_cols:
            return float(self.global_values[level])

        supplied = {"month": month, "hour": hour,
                    "dayofyear": dayofyear, "weekday": weekday}
        missing = [c for c in self.group_cols if supplied[c] is None]
        if missing:
            raise ValueError(f"group columns {missing} need a value at apply time")
        arrays = {c: np.asarray(supplied[c]) for c in self.group_cols}

        size = max(a.size for a in arrays.values())
        shape = np.broadcast_shapes(*[a.shape for a in arrays.values()])
        flat = [np.broadcast_to(a, shape).ravel() for a in arrays.values()]
        out = np.empty(int(np.prod(shape)), dtype=float)
        for i in range(out.size):
            key = tuple(int(a[i]) for a in flat)
            out[i] = self._value_for_key(level, key)
        out = out.reshape(shape)
        if out.size == 1:
            return float(out.ravel()[0])
        return out

    def scenario(self, p50, *, month=None, hour=None, dayofyear=None,
                 weekday=None, levels: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
        """Convert a P50 series into scenario series (non-negative, monotone)."""
        p50 = np.asarray(p50, dtype=float)
        levels = tuple(levels if levels is not None else self.levels)
        out: Dict[int, np.ndarray] = {}
        for level in levels:
            value = self.value(level, month=month, hour=hour,
                               dayofyear=dayofyear, weekday=weekday)
            value = np.asarray(value, dtype=float)
            if self.method == "multiplicative":
                scenario = p50 * value
            else:
                scenario = p50 + value
            out[int(level)] = np.clip(scenario, 0.0, None)

        # Enforce P90 <= P75 <= P50 even when shrinkage or clipping bends an
        # individual group.  This is a physical requirement, not cosmetics: a
        # "conservative" case that is above the central case is a bug.
        ordered = sorted(out)
        for previous, current in zip(ordered, ordered[1:]):
            out[current] = np.minimum(out[current], out[previous])
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "levels": list(self.levels),
            "method": self.method,
            "group_cols": list(self.group_cols),
            "min_samples": self.min_samples,
            "shrinkage": self.shrinkage,
            "calibrate_p50": self.calibrate_p50,
            "global_values": {str(k): v for k, v in self.global_values.items()},
            "group_values": [
                {
                    "key": list(k),
                    "n": self.group_counts.get(k, 0),
                    "values": {str(level): v for level, v in v.items()},
                }
                for k, v in sorted(self.group_values.items())
            ],
        }


@dataclass
class AnnualFactorCalibration:
    """
    Scalar P-value factors derived from the interannual spread of annual energy.

    This is the convention used in renewable-energy finance: P90 is the annual
    energy exceeded in 90 % of years, not the 10th percentile of every hour.
    The factor is applied to the whole P50 series, so the hourly shape is kept
    and only the level is derated.
    """

    target: str
    levels: Tuple[int, ...]
    factors: Dict[int, float]
    annual_by_year: Dict[int, float]
    reference_annual: float
    side: str = "lower"

    def scenario(self, p50, *, month=None, hour=None, dayofyear=None,
                 weekday=None, levels: Optional[Sequence[int]] = None
                 ) -> Dict[int, np.ndarray]:
        p50 = np.asarray(p50, dtype=float)
        levels = tuple(levels if levels is not None else self.levels)
        out: Dict[int, np.ndarray] = {}
        for level in levels:
            out[int(level)] = np.clip(p50 * float(self.factors[int(level)]), 0.0, None)
        # Factors are monotone by construction, but enforce it after clipping so
        # a hand-edited factor file cannot produce P90 > P75.
        ordered = sorted(out)
        for previous, current in zip(ordered, ordered[1:]):
            out[current] = np.minimum(out[current], out[previous])
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "levels": list(self.levels),
            "factors": {str(k): v for k, v in self.factors.items()},
            "annual_by_year": {str(k): v for k, v in self.annual_by_year.items()},
            "reference_annual": self.reference_annual,
            "side": self.side,
        }


def fit_annual_factors(
    actual,
    years,
    *,
    target: str = "generation",
    levels: Sequence[int] = DEFAULT_LEVELS,
    side: str = "lower",
    min_years: int = 3,
) -> AnnualFactorCalibration:
    """
    Derive P75/P90 factors from annual energy totals.

    ``factor_L = Q_(1-L/100)(annual energy) / median(annual energy)`` for the
    lower side, so P50 has factor 1 by construction and P90 is the 10th
    percentile of the annual distribution.  The median, not the mean, is the
    denominator because P50 is a median statement.
    """
    frame = pd.DataFrame({
        "year": np.asarray(years),
        "actual": _as_float(actual),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["actual"] >= 0.0]
    annual = frame.groupby("year")["actual"].sum().sort_index()
    if len(annual) < int(min_years):
        raise ValueError(
            f"annual calibration needs at least {min_years} complete years; "
            f"got {len(annual)}"
        )
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    reference = float(annual.median())
    factors: Dict[int, float] = {}
    for level in (int(v) for v in levels):
        if level == 50:
            factors[level] = 1.0
            continue
        q = (100.0 - level) / 100.0 if side == "lower" else level / 100.0
        value = float(annual.quantile(q))
        factors[level] = value / reference if reference > 0.0 else 1.0
    return AnnualFactorCalibration(
        target=target,
        levels=tuple(int(v) for v in levels),
        factors=factors,
        annual_by_year={int(k): float(v) for k, v in annual.items()},
        reference_annual=reference,
        side=side,
    )


def fit_exceedance_calibration(
    actual,
    p50,
    *,
    target: str = "generation",
    levels: Sequence[int] = DEFAULT_LEVELS,
    group_cols: Sequence[str] = (),
    method: str = "additive",
    min_samples: int = 200,
    shrinkage: float = 50.0,
    zero_floor: float = 1e-6,
    calibrate_p50: bool = False,
    month=None,
    hour=None,
    dayofyear=None,
    weekday=None,
) -> ExceedanceCalibration:
    """
    Fit residual quantiles on a calibration sample.

    ``method='additive'`` fits ``actual - p50``.  ``method='multiplicative'``
    fits ``actual / p50`` and is only attempted where ``p50 > zero_floor``.
    """
    method = str(method).lower()
    if method not in ("additive", "multiplicative"):
        raise ValueError("method must be 'additive' or 'multiplicative'")
    group_cols = tuple(group_cols)
    for col in group_cols:
        if col not in VALID_GROUPS:
            raise ValueError(f"unsupported group column '{col}'; use {VALID_GROUPS}")
    levels = tuple(int(v) for v in levels)

    frame = pd.DataFrame({
        "actual": _as_float(actual),
        "p50": _as_float(p50),
    })
    if len(frame) == 0:
        raise ValueError("cannot calibrate on an empty sample")
    for name, values in (("month", month), ("hour", hour),
                         ("dayofyear", dayofyear), ("weekday", weekday)):
        if name in group_cols:
            if values is None:
                raise ValueError(f"group column '{name}' needs values at fit time")
            frame[name] = np.asarray(values)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actual", "p50", *group_cols]
    )
    frame = frame[(frame["actual"] >= 0.0) & (frame["p50"] >= 0.0)]
    if method == "multiplicative":
        frame = frame[frame["p50"] > float(zero_floor)]
    if len(frame) < 2:
        raise ValueError("fewer than two usable rows after cleaning")

    if method == "multiplicative":
        frame["value"] = frame["actual"] / frame["p50"]
        neutral = 1.0
    else:
        frame["value"] = frame["actual"] - frame["p50"]
        neutral = 0.0

    fit_levels = tuple(l for l in levels if l != 50 or calibrate_p50)
    global_values: Dict[int, float] = {}
    for level in levels:
        if level == 50 and not calibrate_p50:
            global_values[level] = neutral
        else:
            global_values[level] = _quantile(
                frame["value"].to_numpy(), (100.0 - level) / 100.0
            )

    group_values: Dict[Tuple[int, ...], Dict[int, float]] = {}
    group_counts: Dict[Tuple[int, ...], int] = {}
    if group_cols:
        keys = _group_key_frame(frame, group_cols)
        frame = frame.assign(_key=keys)
        for key, sub in frame.groupby("_key", sort=False):
            key = tuple(key) if isinstance(key, tuple) else (key,)
            n = int(len(sub))
            group_counts[key] = n
            if n < int(min_samples):
                continue
            weight = n / (n + float(shrinkage)) if shrinkage > 0 else 1.0
            values: Dict[int, float] = {}
            for level in levels:
                if level == 50 and not calibrate_p50:
                    values[level] = neutral
                else:
                    raw = _quantile(sub["value"].to_numpy(),
                                    (100.0 - level) / 100.0)
                    values[level] = weight * raw + (1.0 - weight) * global_values[level]
            group_values[key] = values

    if group_cols and not group_values:
        print(f"    WARNING: no {', '.join(group_cols)} group reached min_samples={min_samples}; "
              f"all levels fall back to global calibration. Lower --min-samples to use per-group offsets.")

    return ExceedanceCalibration(
        target=target,
        levels=levels,
        method=method,
        group_cols=group_cols,
        global_values=global_values,
        group_values=group_values,
        group_counts=group_counts,
        min_samples=int(min_samples),
        shrinkage=float(shrinkage),
        zero_floor=float(zero_floor),
        calibrate_p50=bool(calibrate_p50),
    )


# ---------------------------------------------------------------------------
# joint solar + wind calibration
# ---------------------------------------------------------------------------
def apply_joint_scenarios(
    solar_p50,
    wind_p50,
    calibration: ExceedanceCalibration,
    *,
    month=None,
    hour=None,
    dayofyear=None,
    weekday=None,
    levels: Optional[Sequence[int]] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply a calibration fitted on *total* solar+wind generation.

    Returns ``{level: (solar, wind, total)}``.  The total scenario is split in
    proportion to the two P50 series, so the sum is preserved and neither
    component can exceed its central value when the level is a downside case.
    """
    solar_p50 = np.asarray(solar_p50, dtype=float)
    wind_p50 = np.asarray(wind_p50, dtype=float)
    if solar_p50.shape != wind_p50.shape:
        raise ValueError("solar and wind P50 arrays must have the same shape")
    total_p50 = solar_p50 + wind_p50
    total_scenarios = calibration.scenario(
        total_p50, month=month, hour=hour, dayofyear=dayofyear,
        weekday=weekday, levels=levels,
    )
    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for level, total in total_scenarios.items():
        share = np.divide(
            solar_p50, total_p50,
            out=np.zeros_like(total_p50), where=total_p50 > 0.0,
        )
        solar = total * share
        wind = total - solar
        out[int(level)] = (np.clip(solar, 0.0, None),
                           np.clip(wind, 0.0, None),
                           np.clip(total, 0.0, None))
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def pinball(actual, scenario, level: int) -> float:
    """Pinball loss for an exceedance level (q = 1 - level/100)."""
    actual = np.asarray(actual, dtype=float)
    scenario = np.asarray(scenario, dtype=float)
    q = (100.0 - int(level)) / 100.0
    diff = actual - scenario
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def score_scenarios(actual, scenarios: Dict[int, np.ndarray], target: str) -> List[Dict[str, Any]]:
    actual = np.asarray(actual, dtype=float)
    rows: List[Dict[str, Any]] = []
    for level in sorted(scenarios):
        scenario = np.asarray(scenarios[level], dtype=float)
        ok = np.isfinite(actual) & np.isfinite(scenario)
        if not ok.any():
            continue
        a = actual[ok]
        s = scenario[ok]
        rows.append({
            "target": target,
            "level": f"P{level}",
            "n": int(ok.sum()),
            "mean_actual": float(np.mean(a)),
            "mean_scenario": float(np.mean(s)),
            "mean_bias_actual_minus_scenario": float(np.mean(a - s)),
            "coverage_actual_ge_scenario": float(np.mean(a >= s)),
            "pinball": pinball(a, s, level),
        })
    return rows


def summarize_annual(hourly: pd.DataFrame, levels: Sequence[int],
                     targets: Sequence[str] = ("solar", "wind", "total")) -> pd.DataFrame:
    """
    Compare annual energy of each scenario with the realised annual energy.

    This is the number a P-value actually refers to.  It is reported alongside
    the hourly coverage, because an hourly P90 and an annual P90 are different
    quantities and are often confused in dispatch studies.
    """
    frame = hourly.copy()
    frame["year"] = pd.to_datetime(frame["time"], errors="coerce").dt.year
    rows: List[Dict[str, Any]] = []
    for target in targets:
        actual_col = f"{target}_actual_MWh"
        if actual_col not in frame.columns:
            continue
        for level in levels:
            scen_col = f"{target}_p{level}_MWh"
            if scen_col not in frame.columns:
                continue
            annual = frame.groupby("year")[[actual_col, scen_col]].sum()
            if annual.empty:
                continue
            actual = annual[actual_col].to_numpy(dtype=float)
            scenario = annual[scen_col].to_numpy(dtype=float)
            rows.append({
                "target": target,
                "level": f"P{level}",
                "years": len(annual),
                "annual_actual_MWh": float(actual.sum()),
                "annual_scenario_MWh": float(scenario.sum()),
                "annual_bias_MWh": float((actual - scenario).sum()),
                "annual_coverage_actual_ge_scenario": float(np.mean(actual >= scenario)),
                "scenario_to_actual_ratio": (
                    float(scenario.sum() / actual.sum()) if actual.sum() > 0 else float("nan")
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# battery-ready input files
# ---------------------------------------------------------------------------
def _base_time(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], errors="coerce")
    required = {"Year", "Month", "Day", "Hour"}
    if not required.issubset(frame.columns):
        raise ValueError("base input needs 'timestamp' or Year/Month/Day/Hour")
    return pd.to_datetime(
        {
            "year": pd.to_numeric(frame["Year"], errors="coerce"),
            "month": pd.to_numeric(frame["Month"], errors="coerce"),
            "day": pd.to_numeric(frame["Day"], errors="coerce"),
            "hour": pd.to_numeric(frame["Hour"], errors="coerce"),
        },
        errors="coerce",
    )


def write_battery_inputs(
    base_input: Path,
    scenario_hourly: pd.DataFrame,
    outdir: Path,
    *,
    levels: Sequence[int] = DEFAULT_LEVELS,
    apply_year: Optional[int] = None,
) -> List[Path]:
    """
    Write one BESSTOOL input per scenario level.

    ``scenario_hourly`` must contain ``time`` and the columns produced by the
    CLI: ``solar_p50_MWh``/``wind_p50_MWh`` and the higher-level variants.
    Rows outside the scenario window keep the realised values; the caller is
    warned when that happens, because mixing a forecast window with realised
    hours makes a KPI hard to interpret.
    """
    base = pd.read_csv(base_input) if str(base_input).lower().endswith(".csv") \
        else pd.read_excel(base_input)
    base = base.copy()
    base["_time"] = _base_time(base)
    if base["_time"].isna().any():
        raise ValueError("base input contains invalid timestamps")
    if apply_year is not None:
        base = base[base["_time"].dt.year == int(apply_year)].copy()
    if base.empty:
        raise ValueError(f"no rows in base input for year {apply_year}")

    scen = scenario_hourly.copy()
    scen["time"] = pd.to_datetime(scen["time"], errors="coerce")
    scen = scen.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")

    merged = base.merge(scen, left_on="_time", right_on="time", how="left",
                        suffixes=("", "_scen"))
    matched = int(merged["time"].notna().sum())
    if matched == 0:
        raise ValueError(
            "no timestamp overlap between the base input and the scenario file"
        )
    if matched < len(merged):
        print(f"    WARNING: {len(merged) - matched} base row(s) outside the "
              f"scenario window keep realised generation")

    written: List[Path] = []
    for level in levels:
        s_col = f"solar_p{level}_MWh"
        w_col = f"wind_p{level}_MWh"
        if s_col not in merged.columns or w_col not in merged.columns:
            raise ValueError(f"scenario file has no columns {s_col}/{w_col}")
        out = merged.copy()
        out["solar_MWh"] = out[s_col].where(out[s_col].notna(), out["solar_MWh"])
        out["wind_MWh"] = out[w_col].where(out[w_col].notna(), out["wind_MWh"])
        out = out.drop(columns=["_time", "time"], errors="ignore")
        path = outdir / f"battery_input_p{level}.csv"
        out.to_csv(path, index=False)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_years(text: str) -> Optional[List[int]]:
    if not text or not str(text).strip():
        return None
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def _parse_groups(text: str) -> Tuple[str, ...]:
    if not text or not str(text).strip():
        return ()
    groups = tuple(v.strip().lower() for v in str(text).split(",") if v.strip())
    for group in groups:
        if group not in VALID_GROUPS:
            raise ValueError(f"unsupported group '{group}'; use {VALID_GROUPS}")
    return groups


def _subset_years(frame: pd.DataFrame, years: Optional[Sequence[int]]) -> pd.DataFrame:
    if years is None:
        return frame
    return frame[frame["time"].dt.year.isin(list(years))].copy()


def build_scenario_hourly(
    solar: pd.DataFrame,
    wind: pd.DataFrame,
    solar_cal: ExceedanceCalibration,
    wind_cal: ExceedanceCalibration,
    joint_cal: Optional[ExceedanceCalibration],
    *,
    levels: Sequence[int],
    aggregation: str,
) -> pd.DataFrame:
    """Line up solar/wind and produce the scenario frame for the test window."""
    merged = solar.merge(
        wind, on="time", how="inner", suffixes=("_solar", "_wind")
    )
    if merged.empty:
        raise ValueError("solar and wind files have no overlapping timestamps")
    merged = merged.sort_values("time").reset_index(drop=True)

    month = merged["month_solar"].to_numpy()
    hour = merged["hour_solar"].to_numpy()
    dayofyear = merged["dayofyear_solar"].to_numpy()
    weekday = merged["weekday_solar"].to_numpy()

    solar_p50 = merged["p50_solar"].to_numpy(dtype=float)
    wind_p50 = merged["p50_wind"].to_numpy(dtype=float)

    solar_scen = solar_cal.scenario(
        solar_p50, month=month, hour=hour, dayofyear=dayofyear,
        weekday=weekday, levels=levels,
    )
    wind_scen = wind_cal.scenario(
        wind_p50, month=month, hour=hour, dayofyear=dayofyear,
        weekday=weekday, levels=levels,
    )

    out = pd.DataFrame({
        "time": merged["time"].to_numpy(),
        "solar_actual_MWh": merged["actual_solar"].to_numpy(dtype=float),
        "wind_actual_MWh": merged["actual_wind"].to_numpy(dtype=float),
        "solar_p50_MWh": solar_p50,
        "wind_p50_MWh": wind_p50,
    })
    out["total_actual_MWh"] = out["solar_actual_MWh"] + out["wind_actual_MWh"]
    out["total_p50_MWh"] = out["solar_p50_MWh"] + out["wind_p50_MWh"]

    if aggregation == "joint":
        if joint_cal is None:
            raise ValueError("joint aggregation requested without a joint calibration")
        joint = apply_joint_scenarios(
            solar_p50, wind_p50, joint_cal, month=month, hour=hour,
            dayofyear=dayofyear, weekday=weekday, levels=levels,
        )
        for level in levels:
            s, w, total = joint[int(level)]
            out[f"solar_p{level}_MWh"] = s
            out[f"wind_p{level}_MWh"] = w
            out[f"total_p{level}_MWh"] = total
    else:
        for level in levels:
            out[f"solar_p{level}_MWh"] = solar_scen[int(level)]
            out[f"wind_p{level}_MWh"] = wind_scen[int(level)]
            out[f"total_p{level}_MWh"] = (
                solar_scen[int(level)] + wind_scen[int(level)]
            )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--solar-pred", required=True,
                        help="CSV/XLSX with time, actual and P50 solar columns")
    parser.add_argument("--wind-pred", default="",
                        help="CSV/XLSX with time, actual and P50 wind columns")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--actual-col", default="actual")
    parser.add_argument("--p50-col", default="pred")
    parser.add_argument("--train-years", default="",
                        help="comma list, e.g. 2020,2021,2022; empty = all rows")
    parser.add_argument("--test-years", default="",
                        help="comma list for scoring/output, e.g. 2023,2024")
    parser.add_argument("--levels", default="50,75,90")
    parser.add_argument("--mode", default="hourly", choices=("hourly", "annual"),
                        help="hourly = residual quantiles per hour; annual = "
                             "scalar P-value factors from interannual energy. "
                             "The two give very different P90 values.")
    parser.add_argument("--group", default="month,hour",
                        help="residual quantile groups: month,hour,dayofyear,weekday")
    parser.add_argument("--residual", default="additive",
                        choices=("additive", "multiplicative"))
    parser.add_argument("--aggregation", default="joint",
                        choices=("joint", "individual"))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--annual-min-years", type=int, default=3,
                        help="minimum complete years for --mode annual")
    parser.add_argument("--calibrate-p50", action="store_true",
                        help="also shift P50 to the residual median; default keeps "
                             "the existing point forecast as the central case")
    parser.add_argument("--side", default="lower", choices=("lower", "upper"),
                        help="lower = P90 is the conservative low case; "
                             "upper = P90 is the optimistic high case")
    parser.add_argument("--annual-actual", default="",
                        help="optional long-run BESSTOOL input (Year/Month/Day/Hour "
                             "+ solar_MWh/wind_MWh) used to fit annual P-values. "
                             "Supply 10+ years here when the forecast file only "
                             "covers the test period.")
    parser.add_argument("--annual-solar-col", default="solar_MWh")
    parser.add_argument("--annual-wind-col", default="wind_MWh")
    parser.add_argument("--base-input", default="",
                        help="optional BESSTOOL input to turn into battery files")
    parser.add_argument("--apply-year", type=int, default=None)
    parser.add_argument("--outdir", default="scenario_out")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    levels = tuple(int(v.strip()) for v in args.levels.split(",") if v.strip())
    if args.side == "upper":
        # P90 upper means the 90th percentile, i.e. q=0.90.  The calibration
        # object stores the quantile to use as (100-level)/100, so mirror the
        # levels here: 10 -> 90, 25 -> 75, 50 -> 50.
        levels = tuple(100 - v for v in levels)
        print("    note: 'upper' side selected; P90 is the optimistic high case")
    groups = _parse_groups(args.group)

    print("=" * 72)
    print("quantile_scenarios.py")
    print(f"  levels      : {', '.join('P' + str(v) for v in levels)}")
    print(f"  mode        : {args.mode}")
    print(f"  residual    : {args.residual}")
    print(f"  groups      : {', '.join(groups) or '(global)'}")
    print(f"  aggregation : {args.aggregation}")
    print("=" * 72)

    solar = load_prediction_file(
        Path(args.solar_pred), time_col=args.time_col,
        actual_col=args.actual_col, p50_col=args.p50_col,
    )
    train_years = _parse_years(args.train_years)
    test_years = _parse_years(args.test_years)
    solar_train = _subset_years(solar, train_years)
    solar_test = _subset_years(solar, test_years)

    wind = None
    wind_train = wind_test = None
    if args.wind_pred:
        wind = load_prediction_file(
            Path(args.wind_pred), time_col=args.time_col,
            actual_col=args.actual_col, p50_col=args.p50_col,
        )
        wind_train = _subset_years(wind, train_years)
        wind_test = _subset_years(wind, test_years)

    def fit(frame: pd.DataFrame, name: str) -> Any:
        if args.mode == "annual":
            return fit_annual_factors(
                frame["actual"], frame["time"].dt.year.to_numpy(), target=name,
                levels=levels, side=args.side, min_years=args.annual_min_years,
            )
        return fit_exceedance_calibration(
            frame["actual"], frame["p50"], target=name, levels=levels,
            group_cols=groups, method=args.residual,
            min_samples=args.min_samples, shrinkage=args.shrinkage,
            calibrate_p50=args.calibrate_p50,
            month=frame["month"].to_numpy() if "month" in groups else None,
            hour=frame["hour"].to_numpy() if "hour" in groups else None,
            dayofyear=frame["dayofyear"].to_numpy() if "dayofyear" in groups else None,
            weekday=frame["weekday"].to_numpy() if "weekday" in groups else None,
        )

    print(f"  solar rows  : train {len(solar_train):,} / test {len(solar_test):,}")
    solar_cal = fit(solar_train, "solar_MWh")

    wind_cal = None
    joint_cal = None
    if wind is not None:
        print(f"  wind rows   : train {len(wind_train):,} / test {len(wind_test):,}")
        wind_cal = fit(wind_train, "wind_MWh")
        if args.aggregation == "joint":
            joined_train = solar_train.merge(
                wind_train, on="time", suffixes=("_solar", "_wind")
            )
            total_actual = (joined_train["actual_solar"] + joined_train["actual_wind"]).to_numpy()
            total_p50 = (joined_train["p50_solar"] + joined_train["p50_wind"]).to_numpy()
            if args.mode == "annual":
                joint_cal = fit_annual_factors(
                    total_actual, joined_train["time"].dt.year.to_numpy(),
                    target="solar+wind_MWh", levels=levels, side=args.side,
                    min_years=args.annual_min_years,
                )
            else:
                joint_cal = fit_exceedance_calibration(
                    total_actual, total_p50, target="solar+wind_MWh", levels=levels,
                    group_cols=groups, method=args.residual,
                    min_samples=args.min_samples, shrinkage=args.shrinkage,
                    calibrate_p50=args.calibrate_p50,
                    month=joined_train["month_solar"].to_numpy() if "month" in groups else None,
                    hour=joined_train["hour_solar"].to_numpy() if "hour" in groups else None,
                    dayofyear=joined_train["dayofyear_solar"].to_numpy() if "dayofyear" in groups else None,
                    weekday=joined_train["weekday_solar"].to_numpy() if "weekday" in groups else None,
                )
            print(f"  joint rows  : train {len(joined_train):,}")

    # Annual P-values are a resource statement, not a forecast-error statement.
    # When a long-run actual file is supplied, fit the factors there instead of
    # on the (possibly short) forecast file.  The P50 series being derated is
    # still the forecast series.
    if args.mode == "annual" and args.annual_actual:
        annual_df = pd.read_csv(args.annual_actual) \
            if str(args.annual_actual).lower().endswith(".csv") \
            else pd.read_excel(args.annual_actual)
        annual_df = _time_columns(annual_df, args.time_col)
        annual_years = annual_df["time"].dt.year.to_numpy()
        if args.annual_solar_col not in annual_df.columns:
            raise ValueError(
                f"annual actual file has no '{args.annual_solar_col}' column"
            )
        solar_actual = pd.to_numeric(
            annual_df[args.annual_solar_col], errors="coerce"
        ).fillna(0.0).to_numpy()
        solar_cal = fit_annual_factors(
            solar_actual, annual_years, target="solar_MWh", levels=levels,
            side=args.side, min_years=args.annual_min_years,
        )
        print(f"  annual solar: {args.annual_actual} "
              f"({len(set(annual_years))} year(s))")
        if wind is not None and args.annual_wind_col in annual_df.columns:
            wind_actual = pd.to_numeric(
                annual_df[args.annual_wind_col], errors="coerce"
            ).fillna(0.0).to_numpy()
            wind_cal = fit_annual_factors(
                wind_actual, annual_years, target="wind_MWh", levels=levels,
                side=args.side, min_years=args.annual_min_years,
            )
            if args.aggregation == "joint":
                joint_cal = fit_annual_factors(
                    solar_actual + wind_actual, annual_years,
                    target="solar+wind_MWh", levels=levels, side=args.side,
                    min_years=args.annual_min_years,
                )
            print(f"  annual wind : {args.annual_actual} "
                  f"({len(set(annual_years))} year(s))")
        for name, cal in (("solar", solar_cal), ("wind", wind_cal),
                          ("joint", joint_cal)):
            if cal is not None:
                print(f"    {name:<6} factors "
                      + ", ".join(f"P{k}={v:.4f}" for k, v in sorted(cal.factors.items())))

    # Score on the requested test window (or the whole file when omitted).
    solar_eval = solar_test if test_years is not None else solar
    rows: List[Dict[str, Any]] = []
    solar_scen = solar_cal.scenario(
        solar_eval["p50"].to_numpy(), month=solar_eval["month"].to_numpy(),
        hour=solar_eval["hour"].to_numpy(),
        dayofyear=solar_eval["dayofyear"].to_numpy(),
        weekday=solar_eval["weekday"].to_numpy(), levels=levels,
    )
    rows += score_scenarios(solar_eval["actual"], solar_scen, "solar_MWh")

    if wind is not None and wind_cal is not None:
        wind_eval = wind_test if test_years is not None else wind
        wind_scen = wind_cal.scenario(
            wind_eval["p50"].to_numpy(), month=wind_eval["month"].to_numpy(),
            hour=wind_eval["hour"].to_numpy(),
            dayofyear=wind_eval["dayofyear"].to_numpy(),
            weekday=wind_eval["weekday"].to_numpy(), levels=levels,
        )
        rows += score_scenarios(wind_eval["actual"], wind_scen, "wind_MWh")

    # Build the full hourly scenario frame over the evaluation window.
    if wind is not None and wind_cal is not None:
        solar_for_frame = solar_test if test_years is not None else solar
        wind_for_frame = wind_test if test_years is not None else wind
        hourly = build_scenario_hourly(
            solar_for_frame, wind_for_frame, solar_cal, wind_cal, joint_cal,
            levels=levels, aggregation=args.aggregation,
        )
        for level in levels:
            rows += score_scenarios(
                hourly["total_actual_MWh"],
                {int(level): hourly[f"total_p{level}_MWh"].to_numpy()},
                "solar+wind_MWh",
            )
    else:
        hourly = pd.DataFrame({
            "time": solar_eval["time"].to_numpy(),
            "solar_actual_MWh": solar_eval["actual"].to_numpy(dtype=float),
            "solar_p50_MWh": solar_eval["p50"].to_numpy(dtype=float),
            "wind_actual_MWh": np.zeros(len(solar_eval)),
            "wind_p50_MWh": np.zeros(len(solar_eval)),
            "total_actual_MWh": solar_eval["actual"].to_numpy(dtype=float),
            "total_p50_MWh": solar_eval["p50"].to_numpy(dtype=float),
        })
        for level in levels:
            hourly[f"solar_p{level}_MWh"] = solar_scen[int(level)]
            hourly[f"wind_p{level}_MWh"] = 0.0
            hourly[f"total_p{level}_MWh"] = solar_scen[int(level)]

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "scenario_summary.csv", index=False)
    hourly.to_csv(outdir / "scenario_hourly.csv", index=False)

    annual = summarize_annual(hourly, levels)
    if not annual.empty:
        annual.to_csv(outdir / "annual_summary.csv", index=False)

    calibration = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "solar_file": str(args.solar_pred),
        "wind_file": str(args.wind_pred) or None,
        "train_years": train_years,
        "test_years": test_years,
        "levels": list(levels),
        "mode": args.mode,
        "side": args.side,
        "aggregation": args.aggregation,
        "residual": args.residual,
        "group_cols": list(groups),
        "solar": solar_cal.to_dict(),
        "wind": wind_cal.to_dict() if wind_cal else None,
        "joint": joint_cal.to_dict() if joint_cal else None,
    }
    (outdir / "scenario_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (outdir / "scenario_summary.json").write_text(
        json.dumps({
            "calibration": {
                "mode": args.mode,
                "solar": solar_cal.to_dict(),
                "wind": wind_cal.to_dict() if wind_cal else None,
                "joint": joint_cal.to_dict() if joint_cal else None,
            },
            "hourly_scores": rows,
        }, indent=2), encoding="utf-8"
    )

    print("\nHourly scores (coverage matches the level only in --mode hourly):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if not annual.empty:
        print("\nAnnual energy (the number a P-value usually refers to):")
        print(annual.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    if args.base_input:
        written = write_battery_inputs(
            Path(args.base_input), hourly, outdir, levels=levels,
            apply_year=args.apply_year,
        )
        print("\nBattery inputs:")
        for path in written:
            print("  ", path)

    print("\nwrote:")
    for name in ("scenario_hourly.csv", "scenario_summary.csv",
                 "annual_summary.csv", "scenario_summary.json",
                 "scenario_calibration.json"):
        print("  ", outdir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
