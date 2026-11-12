# -----------------------------
# PART 1 – imports, dataclass, helpers, single-site (shared BESS)
# -----------------------------
import io
import json
import sys
import time
import calendar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional solver backend for the PART 4 optimiser ---
# PuLP is only needed for the MILP/LP optimiser. The rule-based engine
# (PART 1-3) runs exactly as before when PuLP is missing.
try:  # pragma: no cover - import guard
    import pulp as plp

    PULP_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    plp = None
    PULP_AVAILABLE = False

# --- Vayumithra branding ---
VAYUMITHRA_LOGO_PATH = "vayumithra_logo.jpg"  # put this file next to bess_app.py
VAYUMITHRA_WEBSITE = "https://www.vayumithra.com"  # or any URL you want


# -----------------------------
# Parameters and helpers
# -----------------------------
@dataclass
class BESSParams:
    """Parameters for single/split site and multi-year."""

    # Plant capacities for PLF
    solar_cap_mw: float
    wind_cap_mw: float

    # Contract (used for peak demand + penalty)
    contract_mw: float

    # Mode 1: single-site hybrid (shared BESS)
    bess_energy_mwh: float
    pcs_power_mw: float

    # Mode 2: split-site (separate BESS)
    solar_bess_energy_mwh: float = 0.0
    solar_pcs_power_mw: float = 0.0
    wind_bess_energy_mwh: float = 0.0
    wind_pcs_power_mw: float = 0.0

    # BESS parameters
    eta_charge: float = 0.97  # charge efficiency (0–1)
    eta_discharge: float = 0.96  # discharge efficiency (0–1)
    min_soc_pct: float = 0.0
    max_soc_pct: float = 90.0

    # FLAW 4 FIX - initial state of charge at the start of the run, expressed as
    # a percentage of the usable band (min_soc_pct..max_soc_pct). The old
    # implicit assumption was "full" (100 % of the band), which injected free
    # stored energy at t=0 and inflated early delivery. The neutral default is
    # the middle of the band; the app exposes it as a sidebar control.
    initial_soc_pct: float = 50.0

    peak_hours: List[int] = None

    # If None ⇒ unlimited. If numeric (including 0) ⇒ hard cap in MW.
    grid_limit_mw: Optional[float] = None

    # Modes
    mode: str = "single"  # "single" or "split"
    solar_contract_share: float = 0.84  # kept for compatibility

    # Multi-year
    n_years: int = 1
    rte_by_year: Optional[pd.DataFrame] = None  # per‑year RTE (Year → eff 0–1)

    # Battery reset behaviour
    reset_battery_each_year: bool = False  # if False, SoC carries over between years


def parse_peak_hours(s: str) -> List[int]:
    if not s or not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_degradation_profile(uploaded_file) -> pd.Series:
    """
    Load degradation profile: Year_index (1..25) → Cap_factor (0–1).

    Fixes bug: previously year 25 was effectively treated as 0 aging because the
    index was 0–24 and .get(25, 1.0) returned 1.0. Now we enforce 1–25 indexing.
    """
    idx = list(range(1, 26))

    if uploaded_file is None:
        # Default: no degradation, but correctly defined for all 25 years.
        return pd.Series(1.0, index=idx)

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "Cap_factor" not in df.columns:
        if "Cumulative_%_Capacity" in df.columns:
            df["Cap_factor"] = df["Cumulative_%_Capacity"] / 100.0
        else:
            st.warning("Degradation profile has no Cap_factor column; using 1.0.")
            return pd.Series(1.0, index=idx)

    if "Year_index" not in df.columns:
        st.warning("Degradation profile has no Year_index column; using 1..25 with Cap_factor[0].")
        base_val = float(df["Cap_factor"].iloc[0])
        return pd.Series(base_val, index=idx)

    prof = df.set_index("Year_index")["Cap_factor"]

    # If user used 0–24 indexing, shift to 1–25
    if prof.index.min() == 0 and prof.index.max() == 24:
        prof.index = prof.index + 1

    # Reindex to 1..25; fill gaps with forward/backward fill then 1.0 as last
    # resort. `.ffill()` / `.bfill()` replace the `method=` argument, which was
    # deprecated in pandas 2.1 and removed in pandas 3.x.
    prof = prof.reindex(idx)
    if prof.isna().any():
        prof = prof.ffill().bfill().fillna(1.0)

    return prof


def ensure_hour_int(df: pd.DataFrame) -> pd.DataFrame:
    if np.issubdtype(df["Hour"].dtype, np.datetime64):
        df["Hour"] = df["Hour"].dt.hour
    else:
        df["Hour"] = df["Hour"].apply(
            lambda x: x.hour if hasattr(x, "hour") else int(x)
        )
    return df


# -----------------------------
# FIX 4 - input contract validation
# -----------------------------
REQUIRED_INPUT_COLUMNS = ("Year", "Day", "Month", "Hour", "solar_MWh", "wind_MWh")


def validate_input_frame(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Validate the BESSTOOL_INPUT contract and return a cleaned copy.

    Returns (clean_df, errors, warnings). `errors` are fatal: the caller must
    stop and show them. `warnings` are advisory: the run may proceed.

    The original reader performed no validation at all. A missing column
    surfaced as an unhandled exception outside the simulation error handler,
    and duplicated timestamps were accepted silently, producing a
    state-of-charge trajectory that was arithmetically consistent but
    physically meaningless. Each check below exists because that failure mode
    was observed in practice.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        return df, ["The uploaded file is empty."], warnings

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- column presence, case-insensitive --------------------------------
    lookup = {c.lower(): c for c in df.columns}
    for want in REQUIRED_INPUT_COLUMNS:
        if want in df.columns:
            continue
        hit = lookup.get(want.lower())
        if hit is not None:
            df = df.rename(columns={hit: want})
            warnings.append(f"Column '{hit}' was renamed to '{want}'.")
        else:
            errors.append(f"Missing required column: '{want}'.")
    if errors:
        return df, errors, warnings

    # --- numeric coercion --------------------------------------------------
    for col in ("Year", "Month", "Day", "Hour"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int(df[col].isna().sum())
        if bad:
            errors.append(f"Column '{col}' has {bad} non-numeric or empty value(s).")

    for col in ("solar_MWh", "wind_MWh"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int(df[col].isna().sum())
        if bad:
            errors.append(f"Column '{col}' has {bad} non-numeric or empty value(s).")
            continue
        neg = int((df[col] < 0).sum())
        if neg:
            warnings.append(
                f"Column '{col}' has {neg} negative value(s); they were clipped to 0."
            )
            df[col] = df[col].clip(lower=0.0)

    if errors:
        return df, errors, warnings

    # --- range checks ------------------------------------------------------
    if not bool(df["Month"].between(1, 12).all()):
        errors.append("Column 'Month' must be in 1..12.")
    if not bool(df["Day"].between(1, 31).all()):
        errors.append("Column 'Day' must be in 1..31.")
    if not bool(df["Hour"].between(0, 23).all()):
        errors.append("Column 'Hour' must be in 0..23 (0 = midnight).")

    # --- duplicated hour keys ----------------------------------------------
    dup = df.duplicated(subset=["Year", "Month", "Day", "Hour"], keep=False)
    if bool(dup.any()):
        errors.append(
            f"{int(dup.sum())} row(s) share a (Year, Month, Day, Hour) key. "
            "A duplicated hour breaks the state-of-charge recursion."
        )

    # --- calendar validity (rejects 31 February and friends) ---------------
    for (yy, mm, dd) in df[["Year", "Month", "Day"]].drop_duplicates().itertuples(
        index=False, name=None
    ):
        try:
            pd.Timestamp(year=int(yy), month=int(mm), day=int(dd))
        except (ValueError, OverflowError):
            errors.append(
                f"Invalid calendar date: Year={int(yy)} Month={int(mm)} Day={int(dd)}."
            )
            break

    # --- hour coverage ------------------------------------------------------
    counts = df.groupby(["Year", "Month", "Day"]).size()
    irregular = counts[counts != 24]
    if len(irregular):
        warnings.append(
            f"{len(irregular)} day(s) do not have exactly 24 hourly rows "
            f"(min {int(irregular.min())}, max {int(irregular.max())}). "
            "Penalty and PLF figures assume a full 24-hour day."
        )

    if errors:
        return df, errors, warnings

    for col in ("Year", "Month", "Day", "Hour"):
        df[col] = df[col].astype(int)

    # FLAW 7 FIX - enforce chronological order.
    # The dispatch loops walk the frame row by row and carry the state of
    # charge forward. A file exported in a different order (by hour-of-day, by
    # month, shuffled) therefore produced an arithmetically consistent but
    # physically meaningless trajectory, with no warning anywhere. Sorting here
    # fixes every downstream path at once.
    order_key = (
        df["Year"] * 1_000_000
        + df["Month"] * 10_000
        + df["Day"] * 100
        + df["Hour"]
    )
    if not bool(order_key.is_monotonic_increasing):
        df = df.sort_values(
            ["Year", "Month", "Day", "Hour"], kind="mergesort"
        ).reset_index(drop=True)
        warnings.append(
            "Rows were not in chronological order; they have been sorted by "
            "Year, Month, Day, Hour before dispatch."
        )

    return df, errors, warnings


def expand_to_n_years(
    df: pd.DataFrame, n_years: int
) -> Tuple[pd.DataFrame, str]:
    """
    FLAW 3 FIX - honour the "Number of years (repeat same TS)" control.

    The uploaded file is treated as a representative time series. The run is
    exactly ``n_years`` calendar years long, starting at the first year present
    in the file:

    * one input year -> that year is repeated n_years times;
    * more input years than n_years -> only the first n_years are used;
    * fewer input years than n_years -> the input block is repeated cyclically,
      each copy relabelled to the next calendar year.

    29 February is dropped from a copy whose target year is not a leap year, so
    the relabelled frame always describes a valid calendar.
    """
    n_years = max(1, int(n_years))
    work = df.copy()
    work["Year"] = work["Year"].astype(int)

    src_years = sorted(int(y) for y in work["Year"].unique())
    if not src_years:
        return work, ""

    base_year = int(src_years[0])
    target_years = [base_year + k for k in range(n_years)]

    def _span(years: List[int]) -> str:
        return f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"

    if src_years == target_years:
        return work, (
            f"Simulating {n_years} year(s) exactly as supplied "
            f"({_span(target_years)})."
        )

    blocks: List[pd.DataFrame] = []
    dropped_leap_days = 0
    for k, target in enumerate(target_years):
        src = src_years[k % len(src_years)]
        block = work[work["Year"] == src].copy()
        if not calendar.isleap(target):
            leap = (block["Month"] == 2) & (block["Day"] == 29)
            dropped_leap_days += int(leap.sum())
            block = block[~leap]
        block["Year"] = int(target)
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    out = out.sort_values(
        ["Year", "Month", "Day", "Hour"], kind="mergesort"
    ).reset_index(drop=True)

    if n_years < len(src_years):
        used = src_years[:n_years]
        note = (
            f"The input file contains {len(src_years)} year(s) "
            f"({_span(src_years)}) but 'Number of years' is "
            f"{n_years}, so only {_span(used)} will be simulated. "
            "Increase the control to include the remaining years."
        )
    elif n_years > len(src_years):
        note = (
            f"The input file contains {len(src_years)} year(s); the time series "
            f"is repeated to simulate {n_years} year(s) "
            f"({_span(target_years)})."
        )
    else:
        note = (
            f"Simulating {n_years} year(s): {_span(target_years)}."
        )

    if dropped_leap_days:
        note += (
            f" {dropped_leap_days} row(s) of 29 February were dropped because "
            "the repeated calendar year is not a leap year."
        )
    return out, note


def available_generation_scenarios(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """
    Find P50/P75/P90 scenario columns written by quantile_scenarios.py.

    Accepted names are ``solar_p90_MWh`` / ``wind_p90_MWh`` and the
    upper-case variant ``solar_P90_MWh``.  The function returns
    ``{level: {technology: column}}`` and never guesses a column that is not
    present, so a normal BESSTOOL file with only solar_MWh/wind_MWh behaves
    exactly as before.
    """
    found: Dict[int, Dict[str, str]] = {}
    for level in (50, 75, 90):
        cols: Dict[str, str] = {}
        for tech in ("solar", "wind"):
            for candidate in (f"{tech}_p{level}_MWh", f"{tech}_P{level}_MWh"):
                if candidate in df.columns:
                    cols[tech] = candidate
                    break
        if cols:
            found[level] = cols
    return found


def apply_generation_scenario(
    df: pd.DataFrame, level: int
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Replace ``solar_MWh``/``wind_MWh`` with the selected P-value scenario.

    The replacement is a copy, so the uploaded frame and any later export keep
    the original columns.  A missing scenario value (for example outside the
    forecast window) keeps the realised column for that hour; the artifact
    writer fills those rows explicitly and the app caption says so.
    """
    out = df.copy()
    mapping = available_generation_scenarios(out).get(int(level), {})
    for tech, col in mapping.items():
        values = pd.to_numeric(out[col], errors="coerce")
        fallback = pd.to_numeric(out[f"{tech}_MWh"], errors="coerce").fillna(0.0)
        out[f"{tech}_MWh"] = values.where(values.notna(), fallback).clip(lower=0.0)
    return out, mapping


# ---------------------------------------------------------------------------
# Embedded P50 -> P75/P90 conversion (single-file build)
# ---------------------------------------------------------------------------
# Fitted on the joint solar+wind series by
# quantile_scenarios.fit_annual_factors / fit_exceedance_calibration.
# annual factors : 2015-2024 (10 annual energies), fitted per technology
# hourly residuals: forecast residuals over 2020-2022, tested __HOURLY_TEST__
# Embedded so this file works on its own: upload a plain P50 BESSTOOL_INPUT
# and the app can still build P75/P90.
#
# FLAW 5 FIX - the annual conversion used one joint solar+wind factor on each
# technology. Solar and wind have different interannual spreads, so each now
# uses its own fitted factor. The joint factor is retained below only as a
# portfolio diagnostic; the scenario columns no longer use it.
#
# P90 is the 90 % probability-of-exceedance case (the 10th percentile), so the
# result always satisfies P90 <= P75 <= P50.
#
#   annual : P_L(t) = factor_L(tech) * P50_tech(t),  total = solar + wind
#   hourly : P_L(t) = max(0, P50(t) + Q_(1-L/100)(actual - P50 | month, hour))
SCENARIO_ANNUAL_YEARS = "2015-2024"
SCENARIO_HOURLY_YEARS = "2020-2022"
SCENARIO_ANNUAL_FACTORS = {
    "solar": {75: 0.9878387874594349, 90: 0.9735608352205353},
    "wind": {75: 0.969550530703587, 90: 0.9146545165584424},
}
SCENARIO_ANNUAL_FACTORS_JOINT = {
    75: 0.9621571681651128,
    90: 0.9499951070104649,
}
SCENARIO_HOURLY_GLOBAL = {75: -8.237526421865995, 90: -15.149322669917092}
SCENARIO_HOURLY_GROUPS = {
    ( 1,  0): (-8.31787122608786, -15.313289119705995),
    ( 1,  1): (-8.015111575203715, -15.143404470045642),
    ( 1,  2): (-8.198691079567714, -14.851789928101718),
    ( 1,  3): (-8.841088876854476, -13.917395066047511),
    ( 1,  4): (-8.290315534604803, -13.26078012583217),
    ( 1,  5): (-8.245791288022499, -12.953624582856783),
    ( 1,  6): (-7.382693803404662, -12.496039881828072),
    ( 1,  7): (-5.646764934751551, -9.578662320893493),
    ( 1,  8): (-5.827997666558047, -10.128666754987837),
    ( 1,  9): (-7.023160299887982, -12.282129508864674),
    ( 1, 10): (-7.250013349130837, -13.797276251359603),
    ( 1, 11): (-7.443265164692258, -13.156067027089119),
    ( 1, 12): (-6.183573061344282, -13.833959082871466),
    ( 1, 13): (-6.191415198478966, -12.068486191598723),
    ( 1, 14): (-6.191599646693425, -11.923475379267595),
    ( 1, 15): (-5.7413606924944025, -11.075095271871454),
    ( 1, 16): (-5.793905005388388, -9.6861184137186),
    ( 1, 17): (-6.884636497914606, -11.620210258568541),
    ( 1, 18): (-8.936111986901999, -13.679160879365853),
    ( 1, 19): (-9.64762773943243, -15.952320637953038),
    ( 1, 20): (-10.522856915159462, -16.351146362449477),
    ( 1, 21): (-11.133380051321671, -15.48352328099796),
    ( 1, 22): (-10.386109413331667, -15.4835186467776),
    ( 1, 23): (-9.30306134627788, -15.413158521856843),
    ( 2,  0): (-10.00387880049368, -14.724912691869264),
    ( 2,  1): (-9.968835402134383, -14.838397809775724),
    ( 2,  2): (-10.231348254152909, -14.582168564299042),
    ( 2,  3): (-10.534331283650845, -14.64732097280854),
    ( 2,  4): (-10.43398640666535, -14.351295276016515),
    ( 2,  5): (-9.56624034705355, -13.903811179290955),
    ( 2,  6): (-8.630759829115654, -12.975941424815357),
    ( 2,  7): (-6.894008687861994, -10.686124380135823),
    ( 2,  8): (-6.432052642786196, -9.837912480318163),
    ( 2,  9): (-5.950923298884232, -9.354422394550467),
    ( 2, 10): (-5.597921715835003, -10.880621781192445),
    ( 2, 11): (-5.628647039207854, -10.724617399533265),
    ( 2, 12): (-5.28166592843141, -9.5365180801562),
    ( 2, 13): (-5.632369144290356, -11.005471405889788),
    ( 2, 14): (-5.469595822090718, -11.060115944166444),
    ( 2, 15): (-5.59782748400171, -11.163545593292394),
    ( 2, 16): (-6.057987702093733, -11.452251520442623),
    ( 2, 17): (-7.193529644571086, -11.434729289284093),
    ( 2, 18): (-9.471192978549851, -13.414481806246036),
    ( 2, 19): (-10.044718568600873, -14.495089357029947),
    ( 2, 20): (-10.831479839186265, -15.244721038422927),
    ( 2, 21): (-10.674925357368545, -15.581159247361544),
    ( 2, 22): (-10.609587135532475, -15.321026698919159),
    ( 2, 23): (-9.513100968332084, -15.295824392214058),
    ( 3,  0): (-10.97751469145648, -16.42138156689158),
    ( 3,  1): (-11.250387501241066, -15.985476425452273),
    ( 3,  2): (-10.747845726517518, -16.135510984279744),
    ( 3,  3): (-11.113998484098575, -15.449581950940061),
    ( 3,  4): (-10.53870731824544, -14.953290869061453),
    ( 3,  5): (-9.2741239023516, -14.786372514930141),
    ( 3,  6): (-7.797699417229994, -13.09655312336887),
    ( 3,  7): (-6.669269090295959, -11.570741249332423),
    ( 3,  8): (-6.051681450585723, -10.332183580849122),
    ( 3,  9): (-6.776600411206316, -11.9960136539285),
    ( 3, 10): (-6.066030426687492, -12.524165562156245),
    ( 3, 11): (-5.653434258889657, -10.86206951235805),
    ( 3, 12): (-5.372294028107565, -11.05429455717456),
    ( 3, 13): (-5.752774055821841, -12.018084907760104),
    ( 3, 14): (-6.124144954350435, -12.475195393042593),
    ( 3, 15): (-6.638904317941293, -12.301486842418637),
    ( 3, 16): (-6.866460014647396, -10.919541231736485),
    ( 3, 17): (-7.028957860598258, -11.077258631982886),
    ( 3, 18): (-8.812796470941137, -12.852351624120963),
    ( 3, 19): (-10.528992582596361, -13.979379301394772),
    ( 3, 20): (-10.85849056018038, -14.854527616260448),
    ( 3, 21): (-10.75186049068111, -15.442710015661552),
    ( 3, 22): (-11.031596838936927, -16.418114709628142),
    ( 3, 23): (-11.147227473815725, -16.495392011547846),
    ( 4,  0): (-12.832374230458026, -18.72776024109563),
    ( 4,  1): (-11.162078139761956, -17.960330395909416),
    ( 4,  2): (-11.302483890978126, -17.186594813026215),
    ( 4,  3): (-10.814564212304049, -17.40009314928785),
    ( 4,  4): (-11.492438804481978, -15.92618493635004),
    ( 4,  5): (-10.064882458856246, -15.21567281333582),
    ( 4,  6): (-8.50527200925284, -14.528851593194156),
    ( 4,  7): (-7.163159388162038, -13.353392469275363),
    ( 4,  8): (-7.625712675988683, -13.375741927823007),
    ( 4,  9): (-7.988686233697499, -13.724716741781906),
    ( 4, 10): (-6.673602011975034, -13.692948808890705),
    ( 4, 11): (-7.382900828225333, -14.096183197324804),
    ( 4, 12): (-6.663102615511734, -12.994212052768905),
    ( 4, 13): (-8.41687963261592, -14.68762003268904),
    ( 4, 14): (-8.536340158643794, -14.870139725111157),
    ( 4, 15): (-9.71221073293547, -15.616860722089285),
    ( 4, 16): (-8.451858555042897, -14.789168391516938),
    ( 4, 17): (-8.568605278575344, -14.35587913710179),
    ( 4, 18): (-10.018229507269101, -15.921286817873877),
    ( 4, 19): (-12.79389685007812, -17.76391317797608),
    ( 4, 20): (-12.461828745875986, -17.976802666012418),
    ( 4, 21): (-12.922581230900102, -19.99450794516911),
    ( 4, 22): (-13.995752895755514, -20.52453666260511),
    ( 4, 23): (-13.255573617789937, -19.75443432286272),
    ( 5,  0): (-14.684336052476683, -26.097765221637694),
    ( 5,  1): (-14.323961726853979, -27.27433324629278),
    ( 5,  2): (-14.322707703276816, -23.775964738096157),
    ( 5,  3): (-12.871460351839932, -21.766452695106423),
    ( 5,  4): (-12.108898388370969, -19.180191023713597),
    ( 5,  5): (-11.020846679876657, -18.0770853301739),
    ( 5,  6): (-10.541250405002419, -16.54601004808505),
    ( 5,  7): (-9.365392346207718, -18.966546561513923),
    ( 5,  8): (-7.884996978227733, -19.074021032436008),
    ( 5,  9): (-8.665107262682467, -19.527066825116318),
    ( 5, 10): (-9.463083003093125, -18.462064494499998),
    ( 5, 11): (-9.62381551392948, -16.309954795733912),
    ( 5, 12): (-9.677581487316456, -20.080471512727236),
    ( 5, 13): (-12.417292710990035, -19.094326341677938),
    ( 5, 14): (-11.377210326339013, -20.788423617765208),
    ( 5, 15): (-11.512482929785026, -19.19784178653407),
    ( 5, 16): (-10.52227094773767, -18.78604417791115),
    ( 5, 17): (-9.430637124545719, -19.063511412636586),
    ( 5, 18): (-10.267897585503581, -17.48372331904352),
    ( 5, 19): (-12.618538809621242, -21.010775301749838),
    ( 5, 20): (-14.4144304819933, -21.46608263858563),
    ( 5, 21): (-15.727648964894179, -23.392407575217387),
    ( 5, 22): (-16.116050243751083, -24.41634162682401),
    ( 5, 23): (-14.728781356343923, -25.17069120412586),
    ( 6,  0): (-12.054655062276824, -22.779579780576995),
    ( 6,  1): (-13.885669088250602, -22.680878817282704),
    ( 6,  2): (-12.424156699888808, -21.581077748222036),
    ( 6,  3): (-11.76612520811663, -19.3759232823057),
    ( 6,  4): (-11.379475280335216, -18.26966773649525),
    ( 6,  5): (-10.300622304672228, -19.009798394616134),
    ( 6,  6): (-10.667348839274272, -18.255780566582416),
    ( 6,  7): (-10.434655152380916, -19.86732296369629),
    ( 6,  8): (-8.835693672519117, -20.07249275670323),
    ( 6,  9): (-7.219375524045919, -19.57847963584959),
    ( 6, 10): (-8.212069334607081, -18.89146287992395),
    ( 6, 11): (-7.35387758891398, -19.683138565979785),
    ( 6, 12): (-7.656186520191808, -16.97107069083622),
    ( 6, 13): (-9.558231171142959, -20.290430684234728),
    ( 6, 14): (-9.038040842354462, -19.44799652074754),
    ( 6, 15): (-10.10388479237777, -18.625055774018747),
    ( 6, 16): (-10.528206365549579, -18.115009318330976),
    ( 6, 17): (-10.938233714815496, -16.72145214162682),
    ( 6, 18): (-10.985491529692844, -17.572528506130762),
    ( 6, 19): (-12.732710390365545, -17.918836660534353),
    ( 6, 20): (-13.508160107854168, -21.546753523961826),
    ( 6, 21): (-15.463725840423768, -22.644368738922104),
    ( 6, 22): (-13.048346514427466, -21.610563151368282),
    ( 6, 23): (-12.788296162944375, -21.14300809713697),
    ( 7,  0): (-10.081624731085318, -20.823180713130753),
    ( 7,  1): (-10.449238529187165, -21.20557371448391),
    ( 7,  2): (-10.064615786459205, -20.80031232185604),
    ( 7,  3): (-11.464951143332298, -17.722131713694374),
    ( 7,  4): (-9.625069063744421, -16.511895553040965),
    ( 7,  5): (-10.242806896884389, -16.03120961058913),
    ( 7,  6): (-10.158239896952397, -17.62449279121707),
    ( 7,  7): (-10.137533844263308, -18.297937820421538),
    ( 7,  8): (-11.73461827058028, -21.554411461796754),
    ( 7,  9): (-12.29721110418785, -20.79409248425476),
    ( 7, 10): (-11.631982571478662, -23.097268678789803),
    ( 7, 11): (-9.222727618470717, -19.014071073371927),
    ( 7, 12): (-9.55105329033195, -19.19959175725863),
    ( 7, 13): (-8.014259922961994, -21.55049962085529),
    ( 7, 14): (-9.778666144800157, -21.620590463427327),
    ( 7, 15): (-10.838853313258012, -22.7378995286419),
    ( 7, 16): (-10.081258438663893, -19.492396439366107),
    ( 7, 17): (-10.176063688031137, -17.61571330002156),
    ( 7, 18): (-11.122032767197561, -16.227719486353784),
    ( 7, 19): (-11.490442321736738, -18.813303378551325),
    ( 7, 20): (-12.296439103330966, -19.84890528991371),
    ( 7, 21): (-11.241305716496075, -19.927696832066445),
    ( 7, 22): (-10.625002516037156, -20.918932001269297),
    ( 7, 23): (-10.933647306577804, -19.165225341000564),
    ( 8,  0): (-8.452452081225388, -17.17380078228097),
    ( 8,  1): (-8.818981639028829, -15.559008422201389),
    ( 8,  2): (-9.181967332436429, -14.471789093186615),
    ( 8,  3): (-8.396795096024007, -15.18648051986801),
    ( 8,  4): (-7.4714252822887985, -12.62412050596359),
    ( 8,  5): (-7.075016668985407, -12.533424347632547),
    ( 8,  6): (-7.554018235162243, -13.865605209591664),
    ( 8,  7): (-9.266982028616537, -16.109674886677833),
    ( 8,  8): (-9.809509403084418, -17.827195447811754),
    ( 8,  9): (-9.760021857415362, -18.393773043632475),
    ( 8, 10): (-10.402726068588617, -19.236705015894792),
    ( 8, 11): (-11.226003174350495, -19.141797832151667),
    ( 8, 12): (-11.319485779261628, -17.49670180105067),
    ( 8, 13): (-11.850297679338198, -19.787854646415063),
    ( 8, 14): (-10.42173533369009, -17.480686150835993),
    ( 8, 15): (-9.892846157880433, -15.059693817359557),
    ( 8, 16): (-8.599645873431307, -13.48727716098457),
    ( 8, 17): (-8.422793952904081, -12.53102708461572),
    ( 8, 18): (-7.836133548750885, -12.961686351779012),
    ( 8, 19): (-7.764259189462512, -13.21530485046403),
    ( 8, 20): (-7.7645544868664, -13.703387675072777),
    ( 8, 21): (-8.090470252088279, -15.18598375664462),
    ( 8, 22): (-8.340583644772092, -15.907224962925373),
    ( 8, 23): (-8.489789833628919, -16.182600824139197),
    ( 9,  0): (-8.176410138108157, -11.673237557113094),
    ( 9,  1): (-7.925312140111685, -11.830560521464534),
    ( 9,  2): (-8.0160889168797, -11.798005829829279),
    ( 9,  3): (-7.857788620676782, -11.574480041093846),
    ( 9,  4): (-7.589117097516901, -11.100474576000707),
    ( 9,  5): (-7.215989986699856, -11.105492168322861),
    ( 9,  6): (-7.116278881704208, -11.009152857872536),
    ( 9,  7): (-7.327645231099638, -11.479932865344693),
    ( 9,  8): (-7.523880916699305, -13.417203627915592),
    ( 9,  9): (-7.240994482564818, -13.90235151615975),
    ( 9, 10): (-7.23831320247168, -15.439064914303206),
    ( 9, 11): (-7.24733479602723, -15.11343169045086),
    ( 9, 12): (-7.611165431202559, -16.559307693368943),
    ( 9, 13): (-7.62071554821527, -17.149249456702037),
    ( 9, 14): (-8.164632331457174, -15.115566095428292),
    ( 9, 15): (-8.96210168958476, -14.847219558209273),
    ( 9, 16): (-7.102131089277309, -13.880298431215433),
    ( 9, 17): (-6.393500176679067, -12.73391781174809),
    ( 9, 18): (-6.963562267135129, -11.844484258291939),
    ( 9, 19): (-7.618409935766715, -12.32145269823419),
    ( 9, 20): (-7.740926752370296, -13.04203506039659),
    ( 9, 21): (-7.94193477980729, -13.076404897603293),
    ( 9, 22): (-8.31339363926798, -12.61819688421578),
    ( 9, 23): (-8.335103726045771, -12.494827330707853),
    (10,  0): (-8.296617017895725, -12.725012656630312),
    (10,  1): (-8.185641655599461, -12.504226942492691),
    (10,  2): (-8.065306494382993, -12.412336711389898),
    (10,  3): (-7.901860067496482, -11.864756638641463),
    (10,  4): (-7.560821727214822, -11.670496062568152),
    (10,  5): (-7.398611690327931, -10.872909682495969),
    (10,  6): (-6.711051335631375, -10.475632121151223),
    (10,  7): (-6.409588062827397, -9.881303201667667),
    (10,  8): (-5.482735907725649, -10.099720322449539),
    (10,  9): (-5.35096987397595, -10.07198835286166),
    (10, 10): (-5.312377390039089, -10.783896006846508),
    (10, 11): (-5.018565581940001, -10.942248603635603),
    (10, 12): (-4.461283787021006, -9.362671299993972),
    (10, 13): (-4.997577075145259, -11.30952297887439),
    (10, 14): (-5.968364081901259, -11.528814250091369),
    (10, 15): (-5.893004658258468, -10.955333629624915),
    (10, 16): (-5.851476299227635, -9.454362349421693),
    (10, 17): (-5.6831432366374255, -9.988453632023804),
    (10, 18): (-6.713839941631583, -11.418180726073594),
    (10, 19): (-7.417723587312786, -13.484026640333386),
    (10, 20): (-8.070283953173885, -12.800434929168162),
    (10, 21): (-8.311550352942481, -13.799479273809832),
    (10, 22): (-8.33691619932659, -13.179549232390743),
    (10, 23): (-8.30278533583524, -12.815790772941376),
    (11,  0): (-8.08361784710263, -12.530771990530006),
    (11,  1): (-8.017210867723382, -12.711892048819113),
    (11,  2): (-8.172267622809144, -12.32553994434338),
    (11,  3): (-8.109576599146774, -11.83702554200804),
    (11,  4): (-7.875570300368837, -11.725609118535148),
    (11,  5): (-7.502263656897885, -11.325029664106196),
    (11,  6): (-6.758132807900668, -10.430638378322776),
    (11,  7): (-5.977129420520479, -9.778695104953467),
    (11,  8): (-5.572259604476021, -9.084334242998157),
    (11,  9): (-5.288002794808049, -8.913353035363006),
    (11, 10): (-5.498346258241302, -9.320792698515477),
    (11, 11): (-5.219810256280493, -9.263803838517292),
    (11, 12): (-4.662387095879142, -9.96491648790346),
    (11, 13): (-5.291510512396904, -10.110890433590626),
    (11, 14): (-6.276648499518274, -10.107663327044353),
    (11, 15): (-5.97268625688454, -10.483256347926735),
    (11, 16): (-5.745331177716325, -9.428244232272766),
    (11, 17): (-6.633931043847084, -10.468483685698445),
    (11, 18): (-7.611785041066345, -12.424148922292229),
    (11, 19): (-7.723963360696028, -13.378277973675633),
    (11, 20): (-8.018982533753189, -13.322684962458322),
    (11, 21): (-8.331445486270365, -13.460020991840718),
    (11, 22): (-8.311795772385084, -12.72446841948339),
    (11, 23): (-8.22051797195463, -12.89407153705107),
    (12,  0): (-7.142781560136783, -12.535662708423484),
    (12,  1): (-8.105700620762393, -12.56615251468871),
    (12,  2): (-7.418687064378641, -12.42411195420978),
    (12,  3): (-7.489068008451442, -12.131738116086247),
    (12,  4): (-7.597026798025704, -11.542361506981951),
    (12,  5): (-7.1932117715844335, -11.376134157153423),
    (12,  6): (-6.893032936174478, -11.227560106294781),
    (12,  7): (-5.546488225904664, -9.894003949926496),
    (12,  8): (-5.558421830973719, -10.360006389058885),
    (12,  9): (-5.505831458778159, -11.422465439380286),
    (12, 10): (-5.572064744354539, -11.081392970448025),
    (12, 11): (-4.976220281286239, -9.4758545261034),
    (12, 12): (-5.393638301549466, -11.318065926083737),
    (12, 13): (-5.570169611937207, -10.912985004984739),
    (12, 14): (-5.570927158849891, -11.601691903858198),
    (12, 15): (-5.873179393080553, -10.405221688625645),
    (12, 16): (-5.833037645343346, -9.155289447333026),
    (12, 17): (-6.8244430962545515, -10.213373275928408),
    (12, 18): (-7.411605871068577, -11.509790910393761),
    (12, 19): (-7.868143940473452, -13.138315314888867),
    (12, 20): (-7.974100994817599, -14.006582414608314),
    (12, 21): (-7.973977081203609, -13.472955051234624),
    (12, 22): (-7.416353350667432, -13.25330013605526),
    (12, 23): (-7.089309136448794, -13.368219567078919),
}


def build_generation_scenarios(
    df: pd.DataFrame, mode: str = "annual"
) -> pd.DataFrame:
    """
    Add P50/P75/P90 scenario columns to a plain BESSTOOL_INPUT frame.

    ``mode="annual"`` applies each technology's fitted annual energy factor to
    its own P50 series, keeping the hourly shape (bankability convention), and
    sets the total to their sum. ``mode="hourly"`` adds the month-hour residual
    quantile to the joint total and splits it pro rata on the P50 shares
    (operational stress test). Existing scenario columns are left untouched.
    """
    out = df.copy()
    solar = pd.to_numeric(
        out["solar_p50_MWh"] if "solar_p50_MWh" in out.columns else out["solar_MWh"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    wind = pd.to_numeric(
        out["wind_p50_MWh"] if "wind_p50_MWh" in out.columns else out["wind_MWh"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    total = (solar + wind).to_numpy(dtype=float)
    solar_v = solar.to_numpy(dtype=float)
    wind_v = wind.to_numpy(dtype=float)

    out["solar_p50_MWh"] = solar_v
    out["wind_p50_MWh"] = wind_v
    out["total_p50_MWh"] = total

    if mode == "annual":
        # FLAW 5 FIX - each technology uses its own annual energy factor, and
        # the total is the sum of the two adjusted series. Applying the joint
        # portfolio factor to each technology overstated solar risk and, at
        # P90, understated wind risk.
        for level in (75, 90):
            s = np.clip(
                solar_v * SCENARIO_ANNUAL_FACTORS["solar"][level], 0.0, None
            )
            w = np.clip(
                wind_v * SCENARIO_ANNUAL_FACTORS["wind"][level], 0.0, None
            )
            out[f"solar_p{level}_MWh"] = s
            out[f"wind_p{level}_MWh"] = w
            out[f"total_p{level}_MWh"] = s + w
        # Enforce P90 <= P75 <= P50 per technology even if a factor file is
        # edited by hand (the fitted factors already satisfy this).
        for tech in ("solar", "wind"):
            out[f"{tech}_p90_MWh"] = np.minimum(
                out[f"{tech}_p90_MWh"], out[f"{tech}_p75_MWh"]
            )
        out["total_p90_MWh"] = out["solar_p90_MWh"] + out["wind_p90_MWh"]
        out["total_p75_MWh"] = out["solar_p75_MWh"] + out["wind_p75_MWh"]
        return out

    if "Month" not in out.columns or "Hour" not in out.columns:
        raise ValueError("hourly conversion needs Month and Hour columns")

    month = pd.to_numeric(out["Month"], errors="coerce").fillna(-1).astype(int).tolist()
    hour = pd.to_numeric(out["Hour"], errors="coerce").fillna(-1).astype(int).tolist()
    floor = (SCENARIO_HOURLY_GLOBAL[75], SCENARIO_HOURLY_GLOBAL[90])
    residuals = np.array(
        [SCENARIO_HOURLY_GROUPS.get((m, h), floor) for m, h in zip(month, hour)],
        dtype=float,
    )

    # sequential min enforces P90 <= P75 <= P50 even after clipping
    p75_total = np.clip(total + residuals[:, 0], 0.0, None)
    p75_total = np.minimum(p75_total, total)
    p90_total = np.clip(total + residuals[:, 1], 0.0, None)
    p90_total = np.minimum(p90_total, p75_total)

    share = np.divide(
        solar_v, total, out=np.zeros_like(total), where=total > 0.0
    )
    for level, scen_total in ((75, p75_total), (90, p90_total)):
        scen_solar = np.clip(scen_total * share, 0.0, None)
        out[f"solar_p{level}_MWh"] = scen_solar
        out[f"wind_p{level}_MWh"] = np.clip(scen_total - scen_solar, 0.0, None)
        out[f"total_p{level}_MWh"] = scen_total
    return out


def scenario_columns_present(df: pd.DataFrame) -> bool:
    """True when the upload already carries P75/P90 scenario columns."""
    return bool(available_generation_scenarios(df))


def apply_grid_limit(energy: float, grid_limit_mw: Optional[float]) -> Tuple[float, float]:
    """
    Return (energy_to_grid, curtailed_energy).

    New semantics:
    - grid_limit_mw is None  ⇒ unlimited (no capping)
    - grid_limit_mw >= 0     ⇒ hard cap; 0 means fully blocked (everything curtailed)
    """
    if grid_limit_mw is None:
        return energy, 0.0

    limit = max(grid_limit_mw, 0.0)
    if energy <= limit:
        return energy, 0.0
    return limit, energy - limit


# -----------------------------
# Custom formula profiles
# -----------------------------
PROFILE_PATH = Path("formula_profiles.json")


def load_all_profiles() -> dict:
    """Load all saved formula profiles from JSON."""
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_all_profiles(profiles: dict) -> None:
    """Save all formula profiles to JSON."""
    PROFILE_PATH.write_text(json.dumps(profiles, indent=2))


def apply_custom_formulas(
    hourly: pd.DataFrame,
    formulas_df: pd.DataFrame,
    params: BESSParams,
) -> pd.DataFrame:
    """
    Apply user-defined formulas on top of the hourly dataframe.

    formulas_df must have columns:
        - 'New column'
        - 'Formula'

    Formulas can reference:
        - any existing column in 'hourly'
        - scalar params: contract_mw, eta_charge, eta_discharge
        - numpy as 'np'
    """
    if formulas_df is None or len(formulas_df) == 0:
        return hourly

    df = hourly.copy()

    # Allowed symbols in formulas
    context = {col: df[col] for col in df.columns}
    context.update(
        {
            "contract_mw": params.contract_mw,
            "eta_charge": params.eta_charge,
            "eta_discharge": params.eta_discharge,
            "np": np,
        }
    )

    for _, row in formulas_df.iterrows():
        new_col = str(row.get("New column", "")).strip()
        formula = str(row.get("Formula", "")).strip()
        if not new_col or not formula:
            continue
        try:
            df[new_col] = pd.eval(
                formula,
                local_dict=context,
                engine="python",
            )
            context[new_col] = df[new_col]
        except Exception as e:
            st.warning(f"Could not compute '{new_col}': {e}")

    return df


# -----------------------------
# Dispatch – single shared BESS
# -----------------------------
def simulate_single_site(
    df_in: pd.DataFrame,
    p: BESSParams,
    cap_factor: float,
    initial_soc_mwh: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Mode 1: PV + wind share a single BESS.

    Peak hours:
        * Total plant export (Solar+Wind+BESS) tracks contract and grid limit.
        * No charging during peak.
    Off-peak:
        * BESS charging has first priority from solar+wind.
        * Remaining generation goes to grid, capped by grid_limit_mw if set.
        * Any excess is counted as Curtailed_MWh.
    """
    eta_ch = p.eta_charge
    eta_dis = p.eta_discharge

    df = df_in.copy().reset_index(drop=True)
    df = ensure_hour_int(df)
    df["solar_MWh"] = df["solar_MWh"].astype(float)
    df["wind_MWh"] = df["wind_MWh"].astype(float)
    df["gen_MWh"] = df["solar_MWh"] + df["wind_MWh"]
    df["is_peak"] = df["Hour"].isin(p.peak_hours).astype(int)

    n = len(df)

    eff_bess_cap = p.bess_energy_mwh * cap_factor

    # Basic columns
    df["BESS_Capacity_MWh"] = eff_bess_cap
    df["BESS_SOC_MWh"] = 0.0
    df["BESS_SOC_pct"] = 0.0
    df["BESS_Charge_in_MWh"] = 0.0
    df["BESS_Charge_to_SOC_MWh"] = 0.0
    df["BESS_Charge_Loss_MWh"] = 0.0
    df["BESS_Discharge_MWh"] = 0.0
    df["BESS_TotalCycles"] = 0.0

    # Flows to grid
    df["solar_to_grid_MWh"] = 0.0
    df["wind_to_grid_MWh"] = 0.0

    # Key outputs
    df["GenPlant_to_Grid_MWh"] = 0.0       # solar + wind direct plant export
    df["BESS_to_Grid_MWh"] = 0.0        # BESS discharge to grid
    df["Total_Supplied_MWh"] = 0.0      # Plant_to_Grid + BESS_to_Grid
    df["Peak_Time_Energy_Supplied_MWh"] = 0.0
    df["Offpeak_Supplied_MWh"] = 0.0
    df["Curtailed_MWh"] = 0.0
    df["Penalty_MWh"] = 0.0

    max_energy = eff_bess_cap * (p.max_soc_pct / 100.0)
    min_energy = eff_bess_cap * (p.min_soc_pct / 100.0)

    # Initial SoC: user-configured fraction of the usable band, or a carry-over
    # value clipped to the limits. FLAW 4: this used to default to max_energy.
    if initial_soc_mwh is None:
        soc = min_energy + (float(p.initial_soc_pct) / 100.0) * (
            max_energy - min_energy
        )
    else:
        soc = float(np.clip(initial_soc_mwh, min_energy, max_energy))

    cum_throughput = 0.0

    for i in range(n):
        solar_gen = float(df.at[i, "solar_MWh"])
        wind_gen = float(df.at[i, "wind_MWh"])
        gen = solar_gen + wind_gen
        is_peak = int(df.at[i, "is_peak"])

        charge_in = charge_to_soc = charge_loss = 0.0
        discharge = 0.0
        peak_supplied = offpeak_supplied = 0.0
        penalty = 0.0
        solar_to_grid = 0.0
        wind_to_grid = 0.0
        curtailed = 0.0

        # Effective grid cap (None ⇒ unlimited)
        if p.grid_limit_mw is None:
            grid_cap = None
        else:
            grid_cap = max(p.grid_limit_mw, 0.0)

        if is_peak:
            # ---- PEAK ----
            if grid_cap is None:
                peak_export_target = p.contract_mw
            else:
                peak_export_target = min(p.contract_mw, grid_cap)

            # 1) All generation goes to grid first, capped by peak_export_target
            gen_to_grid, curtailed_from_gen = apply_grid_limit(gen, peak_export_target)
            curtailed += curtailed_from_gen

            # Split generation-to-grid between solar & wind proportionally
            if gen > 0.0 and gen_to_grid > 0.0:
                solar_frac = solar_gen / gen
                wind_frac = wind_gen / gen
                solar_to_grid = gen_to_grid * solar_frac
                wind_to_grid = gen_to_grid * wind_frac

            # 2) BESS discharges to fill remaining shortfall up to peak_export_target
            shortfall = peak_export_target - gen_to_grid
            if shortfall > 0.0 and eff_bess_cap > 0.0:
                avail_discharge_grid = max((soc - min_energy) * eta_dis, 0.0)
                discharge = min(shortfall, avail_discharge_grid, p.pcs_power_mw)
                if discharge > 0.0:
                    soc -= discharge / eta_dis
                    cum_throughput += discharge / eta_dis

            total_export = gen_to_grid + discharge

            # Penalty based on contract_mw
            penalty = max(p.contract_mw - total_export, 0.0)

            peak_supplied = total_export

            plant_to_grid = gen_to_grid

        else:
            # ---- OFF-PEAK ----
            # 1) Charge BESS from generation (first priority)
            remaining_gen = gen
            if eff_bess_cap > 0.0:
                charge_room = max_energy - soc
                if charge_room > 0.0 and remaining_gen > 0.0:
                    possible_charge_in = min(
                        remaining_gen, p.pcs_power_mw, charge_room / eta_ch
                    )
                    if possible_charge_in > 0.0:
                        charge_in = possible_charge_in
                        charge_to_soc = charge_in * eta_ch
                        charge_loss = charge_in - charge_to_soc
                        soc += charge_to_soc
                        cum_throughput += charge_in
                        remaining_gen -= charge_in

            # 2) Remaining gen to grid, capped by grid_limit (if any)
            if grid_cap is None:
                gen_to_grid = remaining_gen
                curtailed_from_gen = 0.0
            else:
                gen_to_grid, curtailed_from_gen = apply_grid_limit(remaining_gen, grid_cap)

            curtailed += curtailed_from_gen

            # Split charging and grid export between solar & wind
            if gen > 0.0:
                solar_frac = solar_gen / gen
                wind_frac = wind_gen / gen
            else:
                solar_frac = wind_frac = 0.0

            # Subtract proportional share of charging from each source
            solar_used_for_charge = charge_in * solar_frac
            wind_used_for_charge = charge_in * wind_frac

            solar_gen_to_grid_potential = max(solar_gen - solar_used_for_charge, 0.0)
            wind_gen_to_grid_potential = max(wind_gen - wind_used_for_charge, 0.0)
            total_potential_to_grid = (
                solar_gen_to_grid_potential + wind_gen_to_grid_potential
            )

            if total_potential_to_grid > 0.0 and gen_to_grid > 0.0:
                solar_share_grid = solar_gen_to_grid_potential / total_potential_to_grid
                wind_share_grid = wind_gen_to_grid_potential / total_potential_to_grid
                solar_to_grid = gen_to_grid * solar_share_grid
                wind_to_grid = gen_to_grid * wind_share_grid

            offpeak_supplied = gen_to_grid

            plant_to_grid = gen_to_grid
            penalty = 0.0

        # ---- Update state ----
        df.at[i, "BESS_SOC_MWh"] = soc
        df.at[i, "BESS_SOC_pct"] = (
            100.0 * soc / eff_bess_cap if eff_bess_cap > 0 else 0.0
        )
        df.at[i, "BESS_Charge_in_MWh"] = charge_in
        df.at[i, "BESS_Charge_to_SOC_MWh"] = charge_to_soc
        df.at[i, "BESS_Charge_Loss_MWh"] = charge_loss
        df.at[i, "BESS_Discharge_MWh"] = discharge
        df.at[i, "solar_to_grid_MWh"] = solar_to_grid
        df.at[i, "wind_to_grid_MWh"] = wind_to_grid

        # direct plant export (Solar+Wind)
        df.at[i, "GenPlant_to_Grid_MWh"] = plant_to_grid
        # battery export
        df.at[i, "BESS_to_Grid_MWh"] = discharge
        # total export
        # FIX 6 - stray local assignment. The original line was
        #   df.at[i, "Total_Supplied_MWh"] = plant_to_Grid = plant_to_grid + discharge
        # which created an unused local named `plant_to_Grid` (capital G) that
        # differs from `plant_to_grid` only in case. Harmless today, but it is
        # exactly the kind of name that a later edit reads by mistake.
        df.at[i, "Total_Supplied_MWh"] = plant_to_grid + discharge

        df.at[i, "Curtailed_MWh"] = curtailed
        df.at[i, "Peak_Time_Energy_Supplied_MWh"] = peak_supplied
        df.at[i, "Offpeak_Supplied_MWh"] = offpeak_supplied
        df.at[i, "Penalty_MWh"] = penalty

        df.at[i, "BESS_TotalCycles"] = (
            cum_throughput / (2.0 * eff_bess_cap) if eff_bess_cap > 0 else 0.0
        )

    return df, soc


# -----------------------------
# PART 2 – split-mode dispatch (solar + wind with separate BESS)
# -----------------------------
def simulate_split_site(
    df_in: pd.DataFrame,
    p: BESSParams,
    cf_solar: float,
    cf_wind: float,
    initial_soc_solar_mwh: Optional[float] = None,
    initial_soc_wind_mwh: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Mode 2: PV+BESS and Wind+BESS separate.

    Peak:
        * Total plant export (solar+wind+BESS) tracked vs contract and grid limit.
        * BESS discharge is split proportional to available SoC.
    Off-peak:
        * Each site charges its own BESS first, then remaining solar+wind goes to grid.
        * Grid export limit is applied on combined solar+wind export.
    """
    eta_ch = p.eta_charge
    eta_dis = p.eta_discharge

    df = df_in.copy().reset_index(drop=True)
    df = ensure_hour_int(df)
    df["solar_MWh"] = df["solar_MWh"].astype(float)
    df["wind_MWh"] = df["wind_MWh"].astype(float)
    df["gen_MWh"] = df["solar_MWh"] + df["wind_MWh"]
    df["is_peak"] = df["Hour"].isin(p.peak_hours).astype(int)

    n = len(df)

    solar_cap_y = p.solar_bess_energy_mwh * cf_solar
    wind_cap_y = p.wind_bess_energy_mwh * cf_wind

    # Columns
    out = df.copy()
    out["solar_bess_capacity_MWh"] = solar_cap_y
    out["wind_bess_capacity_MWh"] = wind_cap_y

    out["solar_bess_soc_MWh"] = 0.0
    out["solar_bess_soc_pct"] = 0.0
    out["solar_bess_charge_in_MWh"] = 0.0
    out["solar_bess_charge_to_soc_MWh"] = 0.0
    out["solar_bess_loss_MWh"] = 0.0
    out["solar_bess_discharge_MWh"] = 0.0
    out["solar_bess_total_cycles"] = 0.0

    out["wind_bess_soc_MWh"] = 0.0
    out["wind_bess_soc_pct"] = 0.0
    out["wind_bess_charge_in_MWh"] = 0.0
    out["wind_bess_charge_to_soc_MWh"] = 0.0
    out["wind_bess_loss_MWh"] = 0.0
    out["wind_bess_discharge_MWh"] = 0.0
    out["wind_bess_total_cycles"] = 0.0

    out["solar_to_grid_MWh"] = 0.0
    out["solar_to_bess_MWh"] = 0.0
    out["wind_to_grid_MWh"] = 0.0
    out["wind_to_bess_MWh"] = 0.0

    out["GenPlant_to_Grid_MWh"] = 0.0      # solar + wind direct export
    out["BESS_to_Grid_MWh"] = 0.0       # solar_bess_discharge + wind_bess_discharge
    out["Total_Supplied_MWh"] = 0.0
    out["Curtailed_MWh"] = 0.0

    out["Peak_Time_Energy_Supplied_MWh"] = 0.0
    out["Offpeak_Supplied_MWh"] = 0.0
    out["Penalty_MWh"] = 0.0

    # BESS capacity ranges
    solar_max_energy = solar_cap_y * (p.max_soc_pct / 100.0)
    solar_min_energy = solar_cap_y * (p.min_soc_pct / 100.0)
    wind_max_energy = wind_cap_y * (p.max_soc_pct / 100.0)
    wind_min_energy = wind_cap_y * (p.min_soc_pct / 100.0)

    if initial_soc_solar_mwh is None:
        solar_soc = solar_min_energy + (float(p.initial_soc_pct) / 100.0) * (
            solar_max_energy - solar_min_energy
        )
    else:
        solar_soc = float(np.clip(initial_soc_solar_mwh, solar_min_energy, solar_max_energy))

    if initial_soc_wind_mwh is None:
        wind_soc = wind_min_energy + (float(p.initial_soc_pct) / 100.0) * (
            wind_max_energy - wind_min_energy
        )
    else:
        wind_soc = float(np.clip(initial_soc_wind_mwh, wind_min_energy, wind_max_energy))

    solar_cum_throughput = 0.0
    wind_cum_throughput = 0.0

    for i in range(n):
        solar_gen = float(out.at[i, "solar_MWh"])
        wind_gen = float(out.at[i, "wind_MWh"])
        total_gen = solar_gen + wind_gen
        is_peak = int(out.at[i, "is_peak"])

        solar_charge_in = solar_charge_to_soc = 0.0
        solar_loss = 0.0
        solar_discharge = 0.0

        wind_charge_in = wind_charge_to_soc = 0.0
        wind_loss = 0.0
        wind_discharge = 0.0

        solar_to_grid = wind_to_grid = 0.0
        solar_to_bess = wind_to_bess = 0.0

        peak_supplied = offpeak_supplied = 0.0
        penalty = 0.0
        curtailed = 0.0

        # Effective grid cap (None ⇒ unlimited)
        if p.grid_limit_mw is None:
            grid_cap = None
        else:
            grid_cap = max(p.grid_limit_mw, 0.0)

        if is_peak:
            # ---- PEAK ----
            # FLAW 6 FIX - the contract is now split between the two
            # technologies with the sidebar-controlled solar share. Each unit is
            # judged against its own target, so a surplus in one cannot hide a
            # shortfall in the other. (Previously the share was hardcoded and,
            # worse, never read: the split engine used one combined contract.)
            if p.grid_limit_mw is None:
                effective_contract = max(float(p.contract_mw), 0.0)
            else:
                effective_contract = min(
                    max(float(p.contract_mw), 0.0),
                    max(float(p.grid_limit_mw), 0.0),
                )
            share = float(np.clip(p.solar_contract_share, 0.0, 1.0))
            solar_target = effective_contract * share
            wind_target = effective_contract * (1.0 - share)

            # 1) Each technology exports its own generation against its own
            #    contract target; generation above that target is curtailed.
            solar_to_grid_from_gen = min(solar_gen, solar_target)
            wind_to_grid_from_gen = min(wind_gen, wind_target)
            gen_to_grid = solar_to_grid_from_gen + wind_to_grid_from_gen
            curtailed += (solar_gen - solar_to_grid_from_gen) + (
                wind_gen - wind_to_grid_from_gen
            )

            # 2) Each BESS discharges into its own residual shortfall only.
            solar_shortfall = max(solar_target - solar_to_grid_from_gen, 0.0)
            wind_shortfall = max(wind_target - wind_to_grid_from_gen, 0.0)

            solar_discharge = max(
                min(
                    solar_shortfall,
                    max((solar_soc - solar_min_energy) * eta_dis, 0.0),
                    max(float(p.solar_pcs_power_mw), 0.0),
                ),
                0.0,
            )
            wind_discharge = max(
                min(
                    wind_shortfall,
                    max((wind_soc - wind_min_energy) * eta_dis, 0.0),
                    max(float(p.wind_pcs_power_mw), 0.0),
                ),
                0.0,
            )

            if solar_discharge > 0.0 and solar_cap_y > 0.0:
                solar_soc -= solar_discharge / eta_dis
                solar_cum_throughput += solar_discharge / eta_dis

            if wind_discharge > 0.0 and wind_cap_y > 0.0:
                wind_soc -= wind_discharge / eta_dis
                wind_cum_throughput += wind_discharge / eta_dis

            # 3) Penalty is the sum of the two residual shortfalls.
            solar_export = solar_to_grid_from_gen + solar_discharge
            wind_export = wind_to_grid_from_gen + wind_discharge
            peak_supplied = solar_export + wind_export
            penalty = max(solar_target - solar_export, 0.0) + max(
                wind_target - wind_export, 0.0
            )

            solar_to_grid = solar_to_grid_from_gen
            wind_to_grid = wind_to_grid_from_gen

            # No charging in peak
            solar_to_bess = 0.0
            wind_to_bess = 0.0

            plant_to_grid = gen_to_grid

        else:
            # ---- OFF-PEAK ----
            remaining_solar = solar_gen
            remaining_wind = wind_gen

            if solar_cap_y > 0.0:
                solar_charge_room = solar_max_energy - solar_soc
                if solar_charge_room > 0.0 and remaining_solar > 0.0:
                    solar_possible_charge = min(
                        remaining_solar,
                        p.solar_pcs_power_mw,
                        solar_charge_room / eta_ch,
                    )
                    if solar_possible_charge > 0.0:
                        solar_charge_in = solar_possible_charge
                        solar_charge_to_soc = solar_charge_in * eta_ch
                        solar_loss = solar_charge_in - solar_charge_to_soc
                        solar_soc += solar_charge_to_soc
                        solar_cum_throughput += solar_charge_in
                        remaining_solar -= solar_charge_in
                        solar_to_bess = solar_charge_in

            if wind_cap_y > 0.0:
                wind_charge_room = wind_max_energy - wind_soc
                if wind_charge_room > 0.0 and remaining_wind > 0.0:
                    wind_possible_charge = min(
                        remaining_wind,
                        p.wind_pcs_power_mw,
                        wind_charge_room / eta_ch,
                    )
                    if wind_possible_charge > 0.0:
                        wind_charge_in = wind_possible_charge
                        wind_charge_to_soc = wind_charge_in * eta_ch
                        wind_loss = wind_charge_in - wind_charge_to_soc
                        wind_soc += wind_charge_to_soc
                        wind_cum_throughput += wind_charge_in
                        remaining_wind -= wind_charge_in
                        wind_to_bess = wind_charge_in

            remaining_total = remaining_solar + remaining_wind

            # Remaining generation to grid, capped by grid limit
            if grid_cap is None:
                gen_to_grid = remaining_total
                curtailed_from_gen = 0.0
            else:
                gen_to_grid, curtailed_from_gen = apply_grid_limit(remaining_total, grid_cap)

            curtailed += curtailed_from_gen

            # Split export between solar & wind based on remaining generation
            if remaining_total > 0.0 and gen_to_grid > 0.0:
                solar_frac = remaining_solar / remaining_total
                wind_frac = remaining_wind / remaining_total
                solar_to_grid = gen_to_grid * solar_frac
                wind_to_grid = gen_to_grid * wind_frac

            total_export = gen_to_grid
            offpeak_supplied = total_export

            plant_to_grid = gen_to_grid
            penalty = 0.0

        # ---- Update per-hour state ----
        out.at[i, "solar_bess_soc_MWh"] = solar_soc
        out.at[i, "solar_bess_soc_pct"] = (
            100.0 * solar_soc / solar_cap_y if solar_cap_y > 0.0 else 0.0
        )
        out.at[i, "solar_bess_charge_in_MWh"] = solar_charge_in
        out.at[i, "solar_bess_charge_to_soc_MWh"] = solar_charge_to_soc
        out.at[i, "solar_bess_loss_MWh"] = solar_loss
        out.at[i, "solar_bess_discharge_MWh"] = solar_discharge

        out.at[i, "wind_bess_soc_MWh"] = wind_soc
        out.at[i, "wind_bess_soc_pct"] = (
            100.0 * wind_soc / wind_cap_y if wind_cap_y > 0.0 else 0.0
        )
        out.at[i, "wind_bess_charge_in_MWh"] = wind_charge_in
        out.at[i, "wind_bess_charge_to_soc_MWh"] = wind_charge_to_soc
        out.at[i, "wind_bess_loss_MWh"] = wind_loss
        out.at[i, "wind_bess_discharge_MWh"] = wind_discharge

        out.at[i, "solar_to_grid_MWh"] = solar_to_grid
        out.at[i, "solar_to_bess_MWh"] = solar_to_bess
        out.at[i, "wind_to_grid_MWh"] = wind_to_grid
        out.at[i, "wind_to_bess_MWh"] = wind_to_bess

        # direct plant export (Solar+Wind)
        out.at[i, "GenPlant_to_Grid_MWh"] = plant_to_grid

        # BESS export
        bess_to_grid = solar_discharge + wind_discharge
        out.at[i, "BESS_to_Grid_MWh"] = bess_to_grid

        # total export
        out.at[i, "Total_Supplied_MWh"] = plant_to_grid + bess_to_grid

        out.at[i, "Curtailed_MWh"] = curtailed

        out.at[i, "Peak_Time_Energy_Supplied_MWh"] = peak_supplied
        out.at[i, "Offpeak_Supplied_MWh"] = offpeak_supplied
        out.at[i, "Penalty_MWh"] = penalty

        out.at[i, "solar_bess_total_cycles"] = (
            solar_cum_throughput / (2.0 * solar_cap_y) if solar_cap_y > 0.0 else 0.0
        )
        out.at[i, "wind_bess_total_cycles"] = (
            wind_cum_throughput / (2.0 * wind_cap_y) if wind_cap_y > 0.0 else 0.0
        )

    return out, solar_soc, wind_soc


# -----------------------------
# Aggregations
# -----------------------------
def make_daily_penalty(df: pd.DataFrame, p: BESSParams) -> pd.DataFrame:
    peak_df = df[df["is_peak"] == 1].copy()
    peak_hours_per_day = len(p.peak_hours)

    group_cols = ["Year", "Month", "Day"]
    daily = (
        peak_df.groupby(group_cols)["Total_Supplied_MWh"]
        .sum()
        .reset_index()
        .rename(columns={"Total_Supplied_MWh": "Peak_supplied_MWh"})
    )

    daily["Required_demand_MWh"] = p.contract_mw * peak_hours_per_day
    daily["Penalty_MWh"] = np.maximum(
        daily["Required_demand_MWh"] - daily["Peak_supplied_MWh"], 0.0
    )

    # --- FIX 1: calendar-safe DATE construction -------------------------
    # Defect: the original code computed `base_year = 2024 + daily["Year"]`
    # unconditionally. The Year column is normally a calendar year (2025), so
    # the exported DATE column came out as 2024 + 2025 = 4049-01-01. The
    # aggregation itself was unaffected, but every date in the daily penalty
    # sheet was wrong, and that sheet is what gets sent to the offtaker.
    #
    # The offset is now applied only when the values actually look like a run
    # index (all values below 100). A file carrying calendar years is used
    # as-is.
    #
    # The DATE value is also assembled as an explicit zero-padded ISO string
    # rather than by passing a dict of components to pd.to_datetime(). Both
    # forms work on pandas 3.0.6, but the dict form is undocumented for this
    # use and the string form is unambiguous about how the parts are joined.
    year_vals = pd.to_numeric(daily["Year"], errors="coerce")
    if year_vals.notna().all() and bool(year_vals.lt(100).all()):
        year_vals = 2024 + year_vals  # index form: 0 -> 2024, 1 -> 2025

    def _iso_part(series: pd.Series, width: int) -> pd.Series:
        return (
            pd.to_numeric(series, errors="coerce")
            .round()
            .astype("Int64")
            .astype(str)
            .str.zfill(width)
        )

    daily["DATE"] = pd.to_datetime(
        _iso_part(year_vals, 4)
        + "-"
        + _iso_part(daily["Month"], 2)
        + "-"
        + _iso_part(daily["Day"], 2),
        format="%Y-%m-%d",
        errors="coerce",
    )

    daily = daily[
        [
            "DATE",
            "Year",
            "Day",
            "Month",
            "Required_demand_MWh",
            "Peak_supplied_MWh",
            "Penalty_MWh",
        ]
    ].sort_values(["Year", "Month", "Day"])

    return daily


def make_monthly_tables(df: pd.DataFrame, p: BESSParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly generation and monthly penalty tables.
    """
    df = df.copy()

    days_per_month = (
        df.groupby(["Year", "Month"])["Day"]
        .nunique()
        .reset_index()
        .rename(columns={"Day": "No_of_Days"})
    )

    agg_dict = {
        "gen_MWh": "sum",
        "solar_MWh": "sum",
        "wind_MWh": "sum",
        "Total_Supplied_MWh": "sum",
        "Peak_Time_Energy_Supplied_MWh": "sum",
        "Offpeak_Supplied_MWh": "sum",
        "Curtailed_MWh": "sum",
        "Penalty_MWh": "sum",
        "GenPlant_to_Grid_MWh": "sum",
        "BESS_to_Grid_MWh": "sum",
    }

    if p.mode == "single":
        agg_dict["BESS_Charge_Loss_MWh"] = "sum"
    else:
        agg_dict["solar_bess_loss_MWh"] = "sum"
        agg_dict["wind_bess_loss_MWh"] = "sum"

    monthly_gen = (
        df.groupby(["Year", "Month"])
        .agg(**{k: (k, v) for k, v in agg_dict.items()})
        .reset_index()
    )

    monthly_gen = monthly_gen.merge(days_per_month, on=["Year", "Month"], how="left")

    monthly_gen = monthly_gen.rename(
        columns={
            "gen_MWh": "Total_Generation_MWh",
            "solar_MWh": "Solar_Generation_MWh",
            "wind_MWh": "Wind_Generation_MWh",
            "Total_Supplied_MWh": "Total_Supplied_MWh",
            "Peak_Time_Energy_Supplied_MWh": "Peak_Supplied_MWh",
            "Offpeak_Supplied_MWh": "Offpeak_Supplied_MWh",
            "Curtailed_MWh": "Curtailed_MWh",
            "Penalty_MWh": "Penalty_MWh",
            "GenPlant_to_Grid_MWh": "GenPlant_to_Grid_MWh",
            "BESS_to_Grid_MWh": "BESS_to_Grid_MWh",
        }
    )

    if p.mode == "single":
        monthly_gen = monthly_gen.rename(
            columns={"BESS_Charge_Loss_MWh": "BESS_Loss_MWh"}
        )
        ordered_cols = [
            "Year",
            "Month",
            "No_of_Days",
            "Solar_Generation_MWh",
            "Wind_Generation_MWh",
            "Total_Generation_MWh",
            "Total_Supplied_MWh",
            "Peak_Supplied_MWh",
            "Offpeak_Supplied_MWh",
            "Curtailed_MWh",
            "Penalty_MWh",
            "GenPlant_to_Grid_MWh",
            "BESS_to_Grid_MWh",
            "BESS_Loss_MWh",
        ]
    else:
        monthly_gen = monthly_gen.rename(
            columns={
                "solar_bess_loss_MWh": "Solar_BESS_Loss_MWh",
                "wind_bess_loss_MWh": "Wind_BESS_Loss_MWh",
            }
        )
        ordered_cols = [
            "Year",
            "Month",
            "No_of_Days",
            "Solar_Generation_MWh",
            "Wind_Generation_MWh",
            "Total_Generation_MWh",
            "Total_Supplied_MWh",
            "Peak_Supplied_MWh",
            "Offpeak_Supplied_MWh",
            "Curtailed_MWh",
            "Penalty_MWh",
            "GenPlant_to_Grid_MWh",
            "BESS_to_Grid_MWh",
            "Solar_BESS_Loss_MWh",
            "Wind_BESS_Loss_MWh",
        ]
    monthly_gen = monthly_gen[ordered_cols].sort_values(["Year", "Month"])

    daily_pen = make_daily_penalty(df, p)
    monthly_pen = (
        daily_pen.copy()
        .assign(Year=lambda x: x["DATE"].dt.year, Month=lambda x: x["DATE"].dt.month)
        .groupby(["Year", "Month"])
        .agg(
            No_of_Days=("DATE", "nunique"),
            Required_demand_MWh=("Required_demand_MWh", "sum"),
            Penalty_MWh=("Penalty_MWh", "sum"),
        )
        .reset_index()
        .sort_values(["Year", "Month"])
    )

    return monthly_gen, monthly_pen

def make_yearly_table(df: pd.DataFrame, p: BESSParams) -> pd.DataFrame:
    rows = []

    for y in sorted(df["Year"].unique()):
        df_y = df[df["Year"] == y]

        # Generation
        solar_gen = df_y["solar_MWh"].sum()
        wind_gen = df_y["wind_MWh"].sum()
        total_gen = solar_gen + wind_gen

        # PLFs
        hours = df_y.shape[0]
        solar_plf = solar_gen / (p.solar_cap_mw * hours) if p.solar_cap_mw > 0 else 0.0
        wind_plf = wind_gen / (p.wind_cap_mw * hours) if p.wind_cap_mw > 0 else 0.0

        hybrid_cap = p.solar_cap_mw + p.wind_cap_mw
        hybrid_plf = total_gen / (hybrid_cap * hours) if hybrid_cap > 0 else 0.0

        row = {
            "Year": y,
            "Solar_PLF_%": solar_plf * 100.0,
            "Wind_PLF_%": wind_plf * 100.0,
            "Hybrid_PLF_%": hybrid_plf * 100.0,
            "Solar_Generation_MWh": solar_gen,
            "Wind_Generation_MWh": wind_gen,
            "Sum_Solar_Wind_Gen_MWh": total_gen,
            "Sum_Total_Supplied_MWh": df_y["Total_Supplied_MWh"].sum(),
            "Sum_GenPlant_to_Grid_MWh": df_y["GenPlant_to_Grid_MWh"].sum(),
            "Sum_BESS_to_Grid_MWh": df_y["BESS_to_Grid_MWh"].sum(),
            "Sum_Curtailed_MWh": df_y["Curtailed_MWh"].sum(),
            "Sum_Peak_Time_Energy_Supplied_MWh": df_y[
                "Peak_Time_Energy_Supplied_MWh"
            ].sum(),
            "Sum_Offpeak_Supplied_MWh": df_y["Offpeak_Supplied_MWh"].sum(),
            "Sum_Penalty_MWh": df_y["Penalty_MWh"].sum(),
        }

        # Add BESS loss columns depending on mode
        if p.mode == "single":
            if "BESS_Charge_Loss_MWh" in df_y.columns:
                row["Sum_BESS_Loss_MWh"] = float(
                    pd.to_numeric(
                        df_y["BESS_Charge_Loss_MWh"], errors="coerce"
                    ).sum()
                )
            else:
                row["Sum_BESS_Loss_MWh"] = 0.0
        else:
            # split mode: solar + wind BESS losses separately
            if "solar_bess_loss_MWh" in df_y.columns:
                row["Sum_Solar_BESS_Loss_MWh"] = float(
                    pd.to_numeric(
                        df_y["solar_bess_loss_MWh"], errors="coerce"
                    ).sum()
                )
            else:
                row["Sum_Solar_BESS_Loss_MWh"] = 0.0

            if "wind_bess_loss_MWh" in df_y.columns:
                row["Sum_Wind_BESS_Loss_MWh"] = float(
                    pd.to_numeric(
                        df_y["wind_bess_loss_MWh"], errors="coerce"
                    ).sum()
                )
            else:
                row["Sum_Wind_BESS_Loss_MWh"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# PART 3 – multi-year controller and Streamlit UI
# -----------------------------
def simulate_multi_year(
    df_base: pd.DataFrame,
    p: BESSParams,
    degr_single: pd.Series,
    degr_solar: pd.Series,
    degr_wind: pd.Series,
) -> pd.DataFrame:
    """
    Multi-year driver with degradation and optional SoC carry-over.
    """
    all_years = []

    shared_soc = None
    solar_soc = None
    wind_soc = None

    years = sorted(df_base["Year"].unique())

    # FLAW 1 + FLAW 2 FIX - the degradation and RTE tables are indexed by
    # project year 1..N, but the driver loop carries calendar years (2024,
    # 2025, ...). Looking a 1..N table up with a calendar year always missed, so
    # degradation silently stayed at 1.0 and year-dependent RTE always fell back
    # to the year-1 values. Map each calendar year to its 1-based position in
    # the simulated run and use that ordinal for both lookups.
    year_ordinal = {int(y): i + 1 for i, y in enumerate(years)}

    for y in years:
        df_y = df_base[df_base["Year"] == y].copy()
        year_index = year_ordinal[int(y)]

        if p.rte_by_year is not None and year_index in p.rte_by_year.index:
            eta_ch_y = float(p.rte_by_year.loc[year_index, "eta_charge"])
            eta_dis_y = float(p.rte_by_year.loc[year_index, "eta_discharge"])
        else:
            eta_ch_y = p.eta_charge
            eta_dis_y = p.eta_discharge

        p_year = BESSParams(
            solar_cap_mw=p.solar_cap_mw,
            wind_cap_mw=p.wind_cap_mw,
            contract_mw=p.contract_mw,
            bess_energy_mwh=p.bess_energy_mwh,
            pcs_power_mw=p.pcs_power_mw,
            solar_bess_energy_mwh=p.solar_bess_energy_mwh,
            solar_pcs_power_mw=p.solar_pcs_power_mw,
            wind_bess_energy_mwh=p.wind_bess_energy_mwh,
            wind_pcs_power_mw=p.wind_pcs_power_mw,
            eta_charge=eta_ch_y,
            eta_discharge=eta_dis_y,
            min_soc_pct=p.min_soc_pct,
            max_soc_pct=p.max_soc_pct,
            initial_soc_pct=p.initial_soc_pct,
            peak_hours=p.peak_hours,
            grid_limit_mw=p.grid_limit_mw,
            mode=p.mode,
            solar_contract_share=p.solar_contract_share,
            n_years=p.n_years,
            rte_by_year=p.rte_by_year,
            reset_battery_each_year=p.reset_battery_each_year,
        )

        if p.mode == "single":
            cf = float(degr_single.get(year_index, 1.0))
            if p.reset_battery_each_year or shared_soc is None:
                init_soc = None
            else:
                init_soc = shared_soc

            hourly_y, shared_soc = simulate_single_site(
                df_y, p_year, cf, initial_soc_mwh=init_soc
            )
        else:
            cf_s = float(degr_solar.get(year_index, 1.0))
            cf_w = float(degr_wind.get(year_index, 1.0))

            if p.reset_battery_each_year or solar_soc is None:
                init_solar_soc = None
            else:
                init_solar_soc = solar_soc

            if p.reset_battery_each_year or wind_soc is None:
                init_wind_soc = None
            else:
                init_wind_soc = wind_soc

            hourly_y, solar_soc, wind_soc = simulate_split_site(
                df_y,
                p_year,
                cf_s,
                cf_w,
                initial_soc_solar_mwh=init_solar_soc,
                initial_soc_wind_mwh=init_wind_soc,
            )

        all_years.append(hourly_y)

    return pd.concat(all_years, ignore_index=True)


# ---------------------------------------------------------------------------
# PART 4 – BESS OPTIMISER (MILP / LP relaxation + rolling horizon)
# ---------------------------------------------------------------------------
# This part is ADDITIVE ONLY. It does not modify any formula, rule, threshold
# or aggregation used by PART 1-3. It produces a dataframe with exactly the
# same schema as `simulate_multi_year`, so `make_daily_penalty`,
# `make_monthly_tables` and `make_yearly_table` can be reused unchanged.
#
# ---------------------------------------------------------------------------
# Design notes – every modelling choice is anchored in external literature
# ---------------------------------------------------------------------------
# [1] STORAGE MODEL ("Exact-MILP").  A single binary per hour is sufficient:
#
#         e[t+1] = rho*e[t] + eta_ch*(c[t] + imp[t]) - d[t]/eta_dis
#         0 <= c[t] + imp[t] <= P_pcs * u[t]
#         0 <= d[t]           <= P_pcs * (1 - u[t])
#         E_min <= e[t] <= E_max
#
#     Sources:
#       * Martin & F. Domin-brand, "Linear Battery Models for Power Systems
#         Analysis", PSCC 2022, arXiv:2204.08240, Model 1 (Exact-MILP):
#           e = e_prev + eta_c*p_c - p_d/eta_d ;  0<=p_c<=P_c*z ;
#           0<=p_d<=P_d*y ; y + z <= 1.
#       * Efkarpidis, Imoscopi & Geidl, "Peak shaving in distribution networks
#         using stationary energy storage systems", Sustainable Energy, Grids
#         and Networks 34 (2023) 101018, Eqs. (17)-(18) – the single-binary
#         form 0<=P_ch<=p(t)*P_nom , 0<=P_dis<=(1-p(t))*P_nom.
#       * Baldick et al., "Battery Storage Formulation and Impact on Security
#         Constrained Unit Commitment", Optimization Online 2021 – proves the
#         single-binary form is both sufficient and tight.
#
# [2] WHEN ARE THE BINARIES ACTUALLY NEEDED?
#     Prat, Lusby & Pinson, "Sufficient A Priori Conditions for the Linear
#     Relaxation of the Energy Storage Scheduling Problem", arXiv:2508.12999:
#       * Proposition 1: if round-trip efficiency eta = eta_ch*eta_dis < 1 AND
#         there are no strictly negative prices, the LP relaxation is EXACT
#         (an optimal solution without simultaneous charge & discharge always
#         exists) -> no binaries required, solve a pure LP.
#       * Corollary 1:  if eta < 1, simultaneity can only appear in strictly
#         negative-price hours -> binaries only there.
#       * Alma massalkhi & Hiskens (2015): with eta = 1 the models coincide.
#     `needs_complementarity()` implements this test, so the app only pays the
#     branch-and-bound cost when the binaries are actually load bearing.
#
# [3] ROLLING HORIZON.  Solve a window, COMMIT only the first `step_hours`,
#     then re-solve with the realised SoC.  Sources:
#       * Elkazaz, Sumner, Pholboon & Thomas, "Microgrid Energy Management
#         Using a Two Stage Rolling Horizon Technique for Controlling an ESS",
#         ICRERA 2018.
#       * Fraunhofer SCAI, "Rolling intrinsic for battery valuation in day-ahead
#         and intraday markets": 48 h horizon, commit the first 24 h.
#       * Fraunhofer ISE / OSTI 1393934: "optimized for 48 hours at a time
#         (T=48). However, only the results of the scheduling period for the
#         first 24 hours are used ... The values of processed energy, remaining
#         capacity, and state of charge after 24 hours are transferred to the
#         next day of the simulation."
#     BUG FIXED HERE: the Solarnex notebook computed `current_soc` inside its
#     rolling loop but never fed it back into `optimize_window`, which pinned
#     `soc[0] == 0.5*capacity` in every window. The SoC is now genuinely
#     propagated, exactly as the references above require.
#
# [4] CYCLE LIFE / THROUGHPUT.  Throughput proxy
#         sum( c + imp + d/eta_dis )  <=  2 * E_eff * cycles_per_year
#     matches the `BESS_TotalCycles = cum_throughput / (2 * E_eff)` definition
#     already used by this tool, and the cycle-budget constraint used in the
#     SCAI rolling-intrinsic paper (constraint (4)) and in OSTI 1393934
#     (total processed energy). Per-window budgets are scaled by
#     `step_hours / hours_in_year` so the annual budget is respected exactly.
#
# [5] TARIFF / SIGNAL PREPROCESSING MUST BE CAUSAL.  The Solarnex
#     `apply_spike_filter` used `rolling(window=5, center=True)`, which peeks
#     two hours into the future of every optimisation window (look-ahead bias)
#     and therefore biases the dispatch upward. Here the filter is strictly
#     trailing (`min_periods=1`, no `center`), and it is applied ONLY to the
#     optional arbitrage tariff - never to the contract-penalty term, because
#     clipping the reliability signal would silently change the contract model.
#
# [6] OBJECTIVE.  Energy cost + demand charge is the standard microgrid BESS
#     objective (Energies 11(6):1371, 2018 – "Optimal Scheduling ... of BESS for
#     Microgrids Considering Contract Demand"; arXiv:2502.08764 – energy cost,
#     peak charge and penalty terms). Here the "demand charge" is expressed in
#     the same currency as the existing tool: the hourly contract shortfall
#     `Penalty_MWh`, plus `Curtailed_MWh`, so optimised runs are directly
#     comparable with the rule-based engine on the KPIs already reported.
# ---------------------------------------------------------------------------

# Solver preference order: CBC ships with PuLP (no external install), then
# HiGHS (via `highspy`), then SCIP / GLPK / system CBC.
_PULP_SOLVER_PREFERENCE: Tuple[str, ...] = (
    "PULP_CBC_CMD",
    "HiGHS",
    "SCIP_CMD",
    "GLPK_CMD",
    "COIN_CMD",
)

HOURS_IN_YEAR = 8760.0

# Simultaneous charge + discharge below this threshold (MWh) is treated as
# solver noise rather than a real degeneracy.
SIMULTANEITY_TOL_MWH = 1e-4


@dataclass
class OptimizerParams:
    """
    Parameters for the MILP / LP-rolling-horizon optimiser.

    Units are kept MWh-based, exactly like the rule-based engine, so that the
    objective weights are directly interpretable.
    """

    # --- Rolling horizon -------------------------------------------------
    horizon_hours: int = 24  # look-ahead window length (one diurnal cycle)
    step_hours: int = 1  # hours committed before re-solving

    # --- Information set (look-ahead / data-leakage control) -------------
    # FLAW 9 FIX.
    # "causal" : each window is solved on a forecast built only from data
    #            available before the window starts. The committed hour is then
    #            settled against realised generation. No future data enters the
    #            decision, so this is an operational (non-anticipative) run.
    # "oracle" : the optimiser sees the realised series for the whole window.
    #            This is the perfect-foresight benchmark / upper bound, not an
    #            operational forecast, and the UI says so.
    forecast_mode: str = "causal"
    # Forecast method used when forecast_mode == "causal":
    #   "seasonal_naive" : same hour on the previous day (24 h lag)
    #   "persistence"    : last observed value for the whole horizon
    forecast_method: str = "seasonal_naive"

    # --- Solver ----------------------------------------------------------
    solver: str = "auto"  # "auto" | "PULP_CBC_CMD" | "HiGHS" | ...
    time_limit_s: float = 60.0
    mip_gap: float = 0.0
    msg: bool = False

    # --- Objective weights (minimise) -----------------------------------
    w_penalty: float = 1000.0  # per MWh of peak shortfall vs contract
    w_curtail: float = 1.0  # per MWh of curtailed generation
    w_degradation: float = 0.0  # per MWh of BESS throughput (wear proxy)
    grid_import_price: float = 0.0  # currency/MWh paid to import for charging

    # Optional arbitrage tariff: {hour_int: price_per_MWh}. When supplied, net
    # export earns `price * MWh`. Disabled when empty.
    tariff: Optional[Dict[int, float]] = None

    # --- Physical / operational switches ---------------------------------
    allow_grid_charging: bool = False  # charge from the grid, not only surplus
    no_charge_in_peak: bool = True  # mirror the rule-based engine
    self_discharge_per_hour: float = 0.0  # rho in [0,1); 0 = no self-discharge
    terminal_soc_pct: Optional[float] = None  # e.g. 90.0; None = free

    # --- advanced engines (stochastic / MPC) ----------------------------
    # Weight on the deviation of the end-of-window SoC from `terminal_soc_pct`,
    # in currency per MWh of deviation. This is the classic MPC terminal cost
    # that stops a receding-horizon controller draining its state at the
    # horizon edge (Rawlings, Mayne & Diehl, "Model Predictive Control", 2012).
    w_terminal_soc: float = 0.0

    # NOTE: the former `emergency_import_multiple` field was removed here.
    # It was declared and exported to the settings sheet but never referenced
    # by the objective or the constraint set, because the two-stage stochastic
    # engine it belonged to was never implemented. A dead knob in the settings
    # sheet is worse than no knob: it invites the user to tune a number that
    # cannot change the schedule.
    max_equivalent_cycles_per_year: Optional[float] = None

    # --- Numerical formulation ------------------------------------------
    # None => auto: binaries are added only when they are provably required
    # (see [2]). True/False force the choice.
    force_complementarity: Optional[bool] = None

    # When the LP relaxation is used, a vanishingly small throughput cost is
    # injected to break the degeneracy that makes the simplex return a vertex
    # with simultaneous charge/discharge. Wang et al. (2023) and Han et al.
    # (2025) use exactly this "penalty in the objective to discourage
    # simultaneous charge and discharge". The magnitude here (1e-6 / MWh) is
    # ~9 orders of magnitude below the default penalty weight, so it only
    # removes ties, it never trades against the objective.
    degeneracy_epsilon: float = 1e-6

    # --- Optional causal spike filter (tariff only) ----------------------
    spike_filter_window: int = 0  # 0 = off, else trailing window length
    spike_filter_k: float = 2.5

    def horizon_signature(self) -> Tuple:
        """Hashable signature used for caching."""
        return (
            self.horizon_hours,
            self.step_hours,
            self.forecast_mode,
            self.forecast_method,
            self.solver,
            round(float(self.time_limit_s), 3),
            round(float(self.mip_gap), 6),
            bool(self.msg),
            round(float(self.w_penalty), 6),
            round(float(self.w_curtail), 6),
            round(float(self.w_degradation), 6),
            round(float(self.grid_import_price), 6),
            tuple(sorted((int(k), float(v)) for k, v in (self.tariff or {}).items())),
            bool(self.allow_grid_charging),
            bool(self.no_charge_in_peak),
            round(float(self.self_discharge_per_hour), 8),
            None if self.terminal_soc_pct is None else round(float(self.terminal_soc_pct), 6),
            float(self.w_terminal_soc),
            None
            if self.max_equivalent_cycles_per_year is None
            else round(float(self.max_equivalent_cycles_per_year), 6),
            None
            if self.force_complementarity is None
            else bool(self.force_complementarity),
            float(self.degeneracy_epsilon),
            int(self.spike_filter_window),
            round(float(self.spike_filter_k), 6),
        )


# -----------------------------
# Solver plumbing
# -----------------------------
def available_optimization_solvers() -> List[str]:
    """Names of the PuLP solvers actually usable on this machine."""
    if not PULP_AVAILABLE:
        return []
    try:
        return list(plp.listSolvers(onlyAvailable=True))
    except Exception:
        return []


def _build_solver(
    solver_name: str,
    time_limit_s: float,
    mip_gap: float,
    msg: bool,
):
    """Instantiate a PuLP solver, degrading kwargs if the backend rejects them."""
    if solver_name == "HiGHS":
        ctor = plp.HiGHS
    elif solver_name == "PULP_CBC_CMD":
        ctor = plp.PULP_CBC_CMD
    elif solver_name == "SCIP_CMD":
        ctor = plp.SCIP_CMD
    elif solver_name == "GLPK_CMD":
        ctor = plp.GLPK_CMD
    elif solver_name == "COIN_CMD":
        ctor = plp.COIN_CMD
    else:
        ctor = plp.PULP_CBC_CMD

    attempts = (
        {"msg": msg, "timeLimit": time_limit_s, "gapRel": mip_gap},
        {"msg": msg, "timeLimit": time_limit_s},
        {"msg": msg},
        {},
    )
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return ctor(**kwargs)
        except Exception as e:  # pragma: no cover - backend specific
            last_err = e
    raise RuntimeError(f"Could not build PuLP solver '{solver_name}': {last_err}")


def resolve_solver(
    solver_name: str = "auto",
    time_limit_s: float = 60.0,
    mip_gap: float = 0.0,
    msg: bool = False,
):
    """
    Return (solver, solver_label). Raises RuntimeError with an actionable
    message when no LP/MILP backend is installed.
    """
    if not PULP_AVAILABLE:
        raise RuntimeError(
            "PuLP is not installed, so the MILP optimiser is unavailable. "
            "Install it with `pip install pulp` (CBC ships with PuLP), or "
            "switch the dispatch engine back to the rule-based mode."
        )

    installed = available_optimization_solvers()

    if solver_name and solver_name != "auto":
        if solver_name not in installed:
            raise RuntimeError(
                f"Solver '{solver_name}' is not available. "
                f"Available on this machine: {installed or 'none'}."
            )
        return _build_solver(solver_name, time_limit_s, mip_gap, msg), solver_name

    for cand in _PULP_SOLVER_PREFERENCE:
        if cand in installed:
            return _build_solver(cand, time_limit_s, mip_gap, msg), cand

    raise RuntimeError(
        "No usable LP/MILP solver found. Install one of: "
        f"{list(_PULP_SOLVER_PREFERENCE)}. (`pip install pulp` provides CBC, "
        "`pip install highspy` provides HiGHS.)"
    )


# -----------------------------
# Signal helpers
# -----------------------------
def apply_causal_spike_filter(
    values: np.ndarray,
    window: int = 5,
    k: float = 2.5,
) -> np.ndarray:
    """
    Causal (trailing) robust clip of an external signal.

    Fixes the look-ahead bias of the Solarnex implementation, which used
    `rolling(window=5, center=True)` and therefore used two hours of future
    information for every point in the optimisation window.

    Bounds are built from the STRICTLY PRIOR `window` observations
    (`shift(1)` before the rolling statistics). Including the current point in
    its own bounds is the classic mistake that lets a large spike inflate its
    own standard deviation and survive the clip.

    Only ever applied to the optional arbitrage tariff - never to the contract
    penalty signal (see design note [5]).
    """
    s = pd.Series(np.asarray(values, dtype=float))
    w = int(window) if window else 0
    if w < 3:
        return s.to_numpy()

    # Causal history: everything strictly before t.
    hist_med = s.shift(1).rolling(window=w, min_periods=2).median()
    hist_std = s.shift(1).rolling(window=w, min_periods=2).std()

    # Backfill the warm-up period with the (still causal) expanding statistics.
    warm_med = s.expanding(min_periods=1).median().shift(1)
    warm_std = s.expanding(min_periods=2).std().shift(1)

    med = hist_med.fillna(warm_med)
    std = hist_std.fillna(warm_std)

    # Degenerate (flat) history -> use a positive floor so the clip stays sane.
    global_std = float(s.std()) if len(s) > 1 else 0.0
    if not np.isfinite(global_std) or global_std <= 0.0:
        return s.to_numpy()
    fallback = pd.Series(global_std, index=s.index)
    std = std.where(std.notna() & (std > 0.0), fallback)

    return s.clip(lower=med - k * std, upper=med + k * std).to_numpy()


def build_tariff_vector(
    hours: np.ndarray,
    tariff: Optional[Dict[int, float]],
    spike_window: int = 0,
    spike_k: float = 2.5,
) -> np.ndarray:
    """Map a {hour_of_day: price} dict onto the window, optionally spike-filtered."""
    if not tariff:
        return np.zeros(len(hours), dtype=float)

    lut = {int(k): float(v) for k, v in tariff.items()}
    raw = np.array([lut.get(int(h), 0.0) for h in hours], dtype=float)

    if spike_window and int(spike_window) >= 3:
        return apply_causal_spike_filter(raw, window=spike_window, k=spike_k)
    return raw


# -----------------------------
# Formulation checks
# -----------------------------
def needs_complementarity(p: BESSParams, o: OptimizerParams) -> Tuple[bool, str]:
    """
    Decide whether the charge/discharge complementarity binaries are required.

    Implements Proposition 1 / Corollary 1 of arXiv:2508.12999 for the
    degenerate case eta = 1, plus a conservative guard whenever any objective
    coefficient is negative (which is exactly the regime where simultaneous
    charging and discharging can become optimal).

    Returns (needs_binaries, human readable reason).
    """
    if o.force_complementarity is not None:
        return bool(o.force_complementarity), "forced by user"

    rte = float(p.eta_charge) * float(p.eta_discharge)

    negative_sources: List[str] = []
    if o.grid_import_price < 0:
        negative_sources.append("negative grid import price")
    if o.tariff and min(o.tariff.values()) < 0:
        negative_sources.append("negative tariff price")

    if negative_sources:
        return True, "negative price signal present (" + ", ".join(negative_sources) + ")"

    if rte >= 1.0 - 1e-12:
        return True, "round-trip efficiency = 1.0 (relaxation not exact)"

    return False, f"RTE = {rte:.4f} < 1 and no negative prices -> LP relaxation is exact"


# -----------------------------
# Window builders
# -----------------------------
def _peak_export_caps(
    peak: np.ndarray,
    gen: np.ndarray,
    contract_mw: float,
    grid_limit_mw: Optional[float],
    big_m: float,
) -> np.ndarray:
    """
    Per-hour cap on (plant export + BESS discharge - grid import).

    Mirrors the rule-based engine exactly:
      * peak  -> min(contract_mw, grid_limit_mw)   (see `simulate_single_site`)
      * offpeak -> grid_limit_mw, or unlimited when grid_limit_mw is None
    """
    caps = np.empty(len(peak), dtype=float)
    grid_cap = None if grid_limit_mw is None else max(float(grid_limit_mw), 0.0)

    for i in range(len(peak)):
        if grid_cap is None:
            caps[i] = contract_mw if peak[i] == 1 else big_m
        else:
            caps[i] = min(contract_mw, grid_cap) if peak[i] == 1 else grid_cap
    return caps



# -----------------------------
# Rolling-horizon drivers
# -----------------------------
def _year_windows(n: int, horizon: int, step: int) -> List[Tuple[int, int, int]]:
    """
    Build (start, stop, commit) windows covering n rows.

    The final window is allowed to be shorter than `horizon`; it commits all of
    its rows so the whole timeline is always covered exactly once.
    """
    horizon = max(int(horizon), 1)
    step = max(min(int(step), horizon), 1)

    windows: List[Tuple[int, int, int]] = []
    start = 0
    while start < n:
        stop = min(start + horizon, n)
        commit = min(step, stop - start)
        windows.append((start, stop, commit))
        start += commit
    return windows


def _causal_forecast_series(
    values: np.ndarray,
    hours: np.ndarray,
    start: int,
    horizon: int,
    method: str = "seasonal_naive",
    prior_values: Optional[np.ndarray] = None,
    prior_hours: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FLAW 9 FIX - forecast `horizon` hours from `start` using only past data.

    The forecast is never allowed to read `values[start:]`. For every lead k it
    uses the same hour-of-day 24 h before the window start (a seasonal-naive
    day-ahead profile, repeated when the horizon exceeds 24 h). During the first
    day of a record there is no previous day inside the frame, so
    `prior_values` / `prior_hours` (the tail of the previous simulated year)
    are used when supplied. If even that is unavailable, the fallback is the
    mean of the same hour over the history already seen, then the mean of the
    whole history, and finally zero (a causal constant prior).

    `persistence` repeats the last observed value for every lead. Both methods
    are causal by construction.
    """
    v = np.asarray(values, dtype=float)
    h_all = np.asarray(hours)
    start = max(int(start), 0)
    out = np.zeros(max(int(horizon), 0), dtype=float)
    if out.size == 0:
        return out

    pv = (
        np.asarray(prior_values, dtype=float)
        if prior_values is not None
        else np.zeros(0, dtype=float)
    )
    ph = (
        np.asarray(prior_hours)
        if prior_hours is not None
        else np.zeros(0, dtype=h_all.dtype)
    )
    if pv.size:
        hist = np.concatenate([pv, v[:start]])
        hist_hours = np.concatenate([ph, h_all[:start]])
    else:
        hist = v[:start]
        hist_hours = h_all[:start]
    if hist.size == 0:
        return out

    if str(method).lower() == "persistence":
        return np.maximum(np.full(out.size, float(hist[-1])), 0.0)

    # `hist` index just past the last observed hour. The same hour one day
    # earlier is therefore `hist.size - 24 + (k % 24)`.
    offset = hist.size

    for k in range(out.size):
        src = offset - 24 + (k % 24)
        if src >= 0:
            out[k] = hist[src]
        else:
            frame_idx = start + k
            h = int(h_all[frame_idx]) if frame_idx < len(h_all) else 0
            mask = hist_hours == h
            if bool(mask.any()):
                out[k] = float(hist[mask].mean())
            elif hist.size:
                out[k] = float(hist.mean())
    return np.maximum(out, 0.0)


def _forecast_window(
    df_full: pd.DataFrame,
    start: int,
    stop: int,
    hours_all: np.ndarray,
    forecast_mode: str,
    forecast_method: str,
    prior_solar: Optional[np.ndarray] = None,
    prior_wind: Optional[np.ndarray] = None,
    prior_hours: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return the window the optimiser is allowed to see (FLAW 9 FIX).

    In ``causal`` mode the generation columns are replaced by a forecast built
    from data strictly before ``start``. In ``oracle`` mode the realised series
    is returned unchanged; that is the perfect-foresight benchmark and is
    labelled as such in the UI.
    """
    win = df_full.iloc[start:stop].copy()
    if str(forecast_mode).lower() != "causal":
        return win
    horizon = stop - start
    win["solar_MWh"] = _causal_forecast_series(
        df_full["solar_MWh"].to_numpy(dtype=float),
        hours_all, start, horizon, forecast_method, prior_solar, prior_hours,
    )
    win["wind_MWh"] = _causal_forecast_series(
        df_full["wind_MWh"].to_numpy(dtype=float),
        hours_all, start, horizon, forecast_method, prior_wind, prior_hours,
    )
    return win


def _progressive_bar(total: int, label: str):
    """Streamlit progress bar that degrades gracefully outside Streamlit."""
    try:
        if not total:
            return None
        return st.progress(0.0, text=label)
    except Exception:  # pragma: no cover
        return None


def _update_bar(bar, done: int, total: int, label: str) -> None:
    if bar is None:
        return
    try:
        bar.progress(min(done / max(total, 1), 1.0), text=label)
    except Exception:  # pragma: no cover
        pass


def _close_bar(bar) -> None:
    if bar is None:
        return
    try:
        bar.empty()
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Degeneracy handling for the LP relaxation
# ---------------------------------------------------------------------------
# Proposition 1 of arXiv:2508.12999 guarantees that when eta < 1 and no negative
# prices exist, the LP relaxation has an SCD-free OPTIMAL SOLUTION. It does NOT
# guarantee that the simplex/CBC will RETURN that particular vertex: when the
# objective is flat (for example when the contract is already met in every peak
# hour, so every penalty term is zero) the optimum is massively degenerate and
# the returned vertex can charge and discharge simultaneously.
#
# This is the well-known "relax and correct" pathology (Almassalkhi & Hiskens,
# "Unclassified Economic Dispatch and Economic...", 2015; see also the survey
# in Prat, Lusby & Pinson 2025, Sec. 2.3). We handle it explicitly:
#   1. tiny simultaneous flows (< SIMULTANEITY_TOL_MWH) are clipped away,
#   2. a real degeneracy triggers an automatic re-solve of that window WITH the
#      complementarity binaries, which is guaranteed physically realisable.
# ---------------------------------------------------------------------------
def _max_simultaneous(charge: np.ndarray, discharge: np.ndarray) -> float:
    if charge.size == 0:
        return 0.0
    return float(np.max(np.minimum(charge, discharge)))


def simulate_optimized_single_site(
    df_in: pd.DataFrame,
    p: BESSParams,
    o: OptimizerParams,
    cap_factor: float,
    initial_soc_mwh: Optional[float] = None,
    progress_total: int = 0,
    progress_bar=None,
    forecast_history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Optimised dispatch, single shared BESS. Returns (hourly, final_soc, diagnostics).

    FLAW 9 FIX - this is a true receding-horizon controller:

    * at every step the optimiser decides on a window built from a *causal*
      forecast (data strictly before the window start), or - in the explicitly
      labelled benchmark mode - from the realised series (perfect foresight);
    * only the committed hours are settled, and they are settled against the
      **realised** generation, not the forecast;
    * the SoC and throughput carried to the next step are the physical state
      after those committed hours, so a bad forecast can never be hidden by the
      next window rewriting history.

    The output schema is byte-for-byte the same as `simulate_single_site`, so the
    existing aggregation functions can consume it without modification.
    """
    solver, solver_label = resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_binaries, bin_reason = needs_complementarity(p, o)

    df = df_in.copy().reset_index(drop=True)
    df = ensure_hour_int(df)
    df["solar_MWh"] = df["solar_MWh"].astype(float)
    df["wind_MWh"] = df["wind_MWh"].astype(float)
    df["gen_MWh"] = df["solar_MWh"] + df["wind_MWh"]
    df["is_peak"] = df["Hour"].isin(p.peak_hours).astype(int)

    hours_all = df["Hour"].to_numpy(dtype=int)
    prior_solar = prior_wind = prior_hours = None
    if forecast_history is not None and len(forecast_history):
        hist = ensure_hour_int(forecast_history.copy())
        prior_solar = hist["solar_MWh"].to_numpy(dtype=float)
        prior_wind = hist["wind_MWh"].to_numpy(dtype=float)
        prior_hours = hist["Hour"].to_numpy(dtype=int)

    n = len(df)
    eff_cap = float(p.bess_energy_mwh) * float(cap_factor)
    max_energy = eff_cap * (p.max_soc_pct / 100.0)
    min_energy = eff_cap * (p.min_soc_pct / 100.0)

    for col, default in [
        ("BESS_Capacity_MWh", 0.0),
        ("BESS_SOC_MWh", 0.0),
        ("BESS_SOC_pct", 0.0),
        ("BESS_Charge_in_MWh", 0.0),
        ("BESS_Charge_to_SOC_MWh", 0.0),
        ("BESS_Charge_Loss_MWh", 0.0),
        ("BESS_Discharge_MWh", 0.0),
        ("BESS_TotalCycles", 0.0),
        ("solar_to_grid_MWh", 0.0),
        ("wind_to_grid_MWh", 0.0),
        ("GenPlant_to_Grid_MWh", 0.0),
        ("BESS_to_Grid_MWh", 0.0),
        ("Total_Supplied_MWh", 0.0),
        ("Peak_Time_Energy_Supplied_MWh", 0.0),
        ("Offpeak_Supplied_MWh", 0.0),
        ("Curtailed_MWh", 0.0),
        ("Penalty_MWh", 0.0),
    ]:
        df[col] = float(default)
    df["BESS_Capacity_MWh"] = eff_cap

    soc = (
        min_energy + (float(p.initial_soc_pct) / 100.0) * (max_energy - min_energy)
        if initial_soc_mwh is None
        else float(np.clip(initial_soc_mwh, min_energy, max_energy))
    )

    windows = _year_windows(n, o.horizon_hours, o.step_hours)
    total_windows = progress_total if progress_total else len(windows)
    cum_throughput = 0.0
    eta_dis_fallback = float(p.eta_discharge)
    eff_cap_fallback = float(p.bess_energy_mwh) * float(cap_factor)

    diag = {
        "solver": solver_label,
        "formulation": "MILP (complementarity binaries)" if use_binaries else "LP (exact relaxation)",
        "complementarity_reason": bin_reason,
        "windows": 0,
        "fallback_windows": 0,
        "degenerate_windows": 0,
        "milp_rescues": 0,
        "solve_seconds": 0.0,
        "objective_first_window": None,
        "last_error": None,
    }

    for wi, (start, stop, commit) in enumerate(windows):
        # FLAW 9 FIX - the optimiser only sees a causal forecast. The realised
        # rows are used later, in _write_single_block, to settle the committed
        # hours.
        win = _forecast_window(
            df, start, stop, hours_all, o.forecast_mode, o.forecast_method,
            prior_solar, prior_wind, prior_hours,
        )

        t0 = time.perf_counter()
        try:
            sol = _optimize_single_window(
                win, p, o, cap_factor, soc, solver, use_binaries
            )
            # Degenerate LP vertex -> re-solve this window with binaries so the
            # committed dispatch is guaranteed physically realisable.
            if sol is not None and not use_binaries:
                sc = _max_simultaneous(sol["charge_in"], sol["discharge"])
                if sc > SIMULTANEITY_TOL_MWH:
                    diag["degenerate_windows"] += 1
                    forced = replace(o, force_complementarity=True)
                    sol_milp = _optimize_single_window(
                        win, p, forced, cap_factor, soc, solver, True
                    )
                    if sol_milp is not None:
                        sol = sol_milp
                        diag["milp_rescues"] += 1
        except Exception as exc:
            sol = None
            diag["last_error"] = f"{type(exc).__name__}: {exc}"
        diag["solve_seconds"] += time.perf_counter() - t0
        diag["windows"] += 1

        if sol is None:
            # Graceful degradation: fall back to the proven rule-based logic.
            # FLAW 9 FIX - run and write ONLY the committed rows. The previous
            # code ran the rule engine over the whole horizon, wrote the whole
            # horizon and advanced the SoC by it, which desynchronised the next
            # window by (horizon - commit) hours.
            diag["fallback_windows"] += 1
            committed = df.iloc[start : start + commit].copy()
            block, soc = simulate_single_site(
                committed, p, cap_factor, initial_soc_mwh=soc
            )
            block = block.reset_index(drop=True)
            # Re-base the cumulative cycle counter on the global throughput.
            # Each row needs its own cumulative value; assigning one scalar to
            # the whole block would report the end-of-block cycle count for
            # every hour whenever commit > 1.
            blk_cum = (
                block["BESS_Charge_in_MWh"].cumsum()
                + block["BESS_Discharge_MWh"].cumsum() / max(eta_dis_fallback, 1e-9)
            )
            if eff_cap_fallback > 0.0:
                block["BESS_TotalCycles"] = (
                    cum_throughput + blk_cum
                ) / (2.0 * eff_cap_fallback)
            cum_throughput += float(blk_cum.iloc[-1]) if len(blk_cum) else 0.0
            for col in df.columns:
                if col in block.columns:
                    df.iloc[
                        start : start + commit, df.columns.get_loc(col)
                    ] = block[col].to_numpy()
        else:
            soc, cum_throughput = _write_single_block(
                df, start, commit, sol, p, cap_factor, soc, cum_throughput,
                oracle=str(o.forecast_mode).lower() != "causal",
            )
            if diag["objective_first_window"] is None:
                diag["objective_first_window"] = sol.get("objective")

        # FLAW 9 FIX - with a 1-hour commit this is called 8760 times per year.
        # Updating every hour dominates the runtime, so throttle it.
        if wi % 24 == 0 or wi == len(windows) - 1:
            _update_bar(
                progress_bar, wi + 1, total_windows,
                f"Optimising year: window {wi + 1}/{len(windows)}",
            )

    return df, soc, diag


def _optimize_single_window(
    win: pd.DataFrame,
    p: BESSParams,
    o: OptimizerParams,
    cap_factor: float,
    soc_start: float,
    solver,
    use_binaries: bool,
) -> Optional[Dict[str, Any]]:
    """Build + solve one MILP/LP window and return the committed solution."""
    T = len(win)
    solar = win["solar_MWh"].to_numpy(dtype=float)
    wind = win["wind_MWh"].to_numpy(dtype=float)
    gen = solar + wind
    peak = win["is_peak"].to_numpy(dtype=int)
    hours = win["Hour"].to_numpy()

    # Degeneracy breaker for the LP relaxation (see OptimizerParams docstring).
    w_throughput = float(o.w_degradation)
    if not use_binaries and w_throughput <= 0.0:
        w_throughput = max(float(o.degeneracy_epsilon), 0.0)

    eff_cap = float(p.bess_energy_mwh) * float(cap_factor)
    e_max = eff_cap * (p.max_soc_pct / 100.0)
    e_min = eff_cap * (p.min_soc_pct / 100.0)
    lo, hi = min(e_min, e_max), max(e_min, e_max)
    e0 = float(np.clip(soc_start, lo, hi))
    p_pcs = max(float(p.pcs_power_mw), 0.0)

    eta_ch = float(p.eta_charge)
    eta_dis = float(p.eta_discharge)
    inv_eta_dis = 1.0 / max(eta_dis, 1e-9)
    rho = 1.0 - max(0.0, min(0.999, float(o.self_discharge_per_hour)))

    big_m = float(np.sum(gen)) + p_pcs + max(abs(p.contract_mw), 0.0) + 1.0
    caps = _peak_export_caps(peak, gen, p.contract_mw, p.grid_limit_mw, big_m)
    tariff = build_tariff_vector(hours, o.tariff, o.spike_filter_window, o.spike_filter_k)

    m = plp.LpProblem("BESS_Dispatch_Single", plp.LpMinimize)

    e = [plp.LpVariable(f"e_{t}", lowBound=e_min, upBound=e_max) for t in range(T + 1)]
    c = [plp.LpVariable(f"c_{t}", lowBound=0.0, upBound=p_pcs) for t in range(T)]
    d = [plp.LpVariable(f"d_{t}", lowBound=0.0, upBound=p_pcs) for t in range(T)]
    g = [plp.LpVariable(f"g_{t}", lowBound=0.0, upBound=float(gen[t])) for t in range(T)]
    imp = [plp.LpVariable(f"imp_{t}", lowBound=0.0, upBound=p_pcs) for t in range(T)]
    pen = [
        # The contract shortfall only exists during peak hours: the rule-based
        # engine hard-codes penalty = 0 off-peak, so the optimiser must not be
        # allowed to "earn" a penalty reduction in off-peak hours (that would
        # make it dump the battery into any off-peak hour it could reach the
        # contract in).
        plp.LpVariable(f"pen_{t}", lowBound=0.0,
                       upBound=max(p.contract_mw, 0.0) if peak[t] == 1 else 0.0)
        for t in range(T)
    ]
    cur = [
        plp.LpVariable(f"cur_{t}", lowBound=0.0, upBound=float(gen[t])) for t in range(T)
    ]
    u = [plp.LpVariable(f"u_{t}", cat="Binary") for t in range(T)] if use_binaries else None

    m += e[0] == e0, "initial_soc"

    for t in range(T):
        m += g[t] + d[t] - imp[t] <= caps[t] + 1e-7, f"export_cap_{t}"
        m += g[t] + c[t] <= gen[t] + 1e-7, f"plant_balance_{t}"
        m += c[t] + imp[t] <= p_pcs + 1e-7, f"pcs_charge_{t}"

        if not o.allow_grid_charging:
            m += imp[t] <= 1e-7, f"no_grid_import_{t}"
        if o.no_charge_in_peak:
            m += c[t] + imp[t] <= p_pcs * (1 - int(peak[t])) + 1e-7, f"no_peak_charge_{t}"

        m += (
            e[t + 1]
            == rho * e[t] + eta_ch * (c[t] + imp[t]) - inv_eta_dis * d[t]
        ), f"soc_{t}"

        if u is not None:
            m += c[t] + imp[t] <= p_pcs * u[t] + 1e-7, f"comp_ch_{t}"
            m += d[t] <= p_pcs * (1 - u[t]) + 1e-7, f"comp_dis_{t}"

        m += cur[t] >= gen[t] - (g[t] + c[t]), f"cur_def_{t}"

        # The shortfall is only *defined* during peak hours. Off-peak hours are
        # pinned to zero by their bounds, so adding this constraint there would
        # wrongly force every off-peak hour to reach the contract.
        if peak[t] == 1:
            m += pen[t] >= p.contract_mw - (g[t] + d[t]), f"pen_def_{t}"

    if o.terminal_soc_pct is not None:
        m += e[T] >= eff_cap * (float(o.terminal_soc_pct) / 100.0) - 1e-7, "terminal_soc"

    if o.max_equivalent_cycles_per_year is not None and eff_cap > 0.0:
        step = max(min(o.step_hours, T), 1)
        budget = (
            2.0
            * eff_cap
            * float(o.max_equivalent_cycles_per_year)
            * (step / HOURS_IN_YEAR)
        )
        thr = plp.LpVariable("throughput", lowBound=0.0)
        m += thr == plp.lpSum(c[t] + imp[t] + inv_eta_dis * d[t] for t in range(T))
        m += thr <= budget + 1e-7, "cycle_budget"

    # FIX 5a - the terminal SoC cost is now actually used.
    # `w_terminal_soc` was declared, hashed into horizon_signature() and
    # exported to the settings sheet, but never entered the objective, so a
    # positive value had no effect on the schedule. It is now the standard
    # MPC terminal cost: a linear penalty on |e[T] - terminal target|, which
    # is what stops a receding-horizon controller draining the battery at the
    # horizon edge (Rawlings, Mayne & Diehl, 2012).
    term_cost = 0.0
    if float(o.w_terminal_soc) > 0.0:
        # The cost must not depend on the hard terminal constraint being set,
        # or it becomes the same kind of dead parameter it replaced. When no
        # target is requested the cost aims at the midpoint of the operating
        # band, which is the neutral choice for a daily-cycling battery.
        _target_pct = (
            float(o.terminal_soc_pct)
            if o.terminal_soc_pct is not None
            else 0.5 * (float(p.min_soc_pct) + float(p.max_soc_pct))
        )
        _target = eff_cap * (_target_pct / 100.0)
        _dev_pos = plp.LpVariable("term_dev_pos", lowBound=0.0)
        _dev_neg = plp.LpVariable("term_dev_neg", lowBound=0.0)
        m += _dev_pos >= e[T] - _target, "term_soc_above"
        m += _dev_neg >= _target - e[T], "term_soc_below"
        term_cost = float(o.w_terminal_soc) * (_dev_pos + _dev_neg)

    m += plp.lpSum(
        o.w_penalty * pen[t]
        + o.w_curtail * cur[t]
        + o.grid_import_price * imp[t]
        + w_throughput * (c[t] + imp[t] + inv_eta_dis * d[t])
        - tariff[t] * (g[t] + d[t] - imp[t])
        for t in range(T)
    ) + term_cost

    status = m.solve(solver)
    status_name = plp.LpStatus.get(status, str(status))
    if status_name not in ("Optimal", "Feasible"):
        return None

    def arr(vars_):
        return np.array(
            [0.0 if v.varValue is None else float(v.varValue) for v in vars_],
            dtype=float,
        )

    charge = np.maximum(arr(c), 0.0) + np.maximum(arr(imp), 0.0)
    discharge = np.maximum(arr(d), 0.0)
    export_gen = np.clip(arr(g), 0.0, np.maximum(gen, 0.0))
    e_traj = np.clip(arr(e), 0.0, None)
    pen_traj = np.maximum(arr(pen), 0.0)
    cur_traj = np.maximum(arr(cur), 0.0)

    # Numerical hygiene: never let solver tolerance push the plant charge above
    # the generation that was actually available for it.
    plant_charge = np.minimum(charge, np.maximum(gen - export_gen, 0.0))
    charge = np.minimum(charge, plant_charge + np.maximum(arr(imp), 0.0))

    return {
        "charge_in": charge,
        "import": np.maximum(arr(imp), 0.0),
        "discharge": discharge,
        "gen_to_grid": export_gen,
        # e[t] is the SoC at the START of hour t, so the end-of-hour value for
        # hour t is e[t+1]. Keeping this convention makes `BESS_SOC_MWh` match
        # `simulate_single_site`, which records the SoC after the hour's action.
        "soc_end": e_traj[1 : T + 1],
        "soc": e_traj,
        "penalty": pen_traj,
        "curtailed": cur_traj,
        "objective": None
        if m.objective is None or m.objective.value() is None
        else float(m.objective.value()),
    }


def _write_single_block(
    df: pd.DataFrame,
    start: int,
    commit: int,
    sol: Dict[str, Any],
    p: BESSParams,
    cap_factor: float,
    soc_before: float,
    cum_throughput_before: float,
    oracle: bool = False,
) -> Tuple[float, float]:
    """
    Settle the committed part of a window against the realised generation.

    FLAW 9 FIX - the optimiser planned on a forecast, so the committed actions
    must be replayed against what actually happened. The plan is used as the
    *intent*; every energy flow is capped by the physical limits of the realised
    hour:

    * charging is capped by the realised generation available after the planned
      plant export and by the PCS / charge room;
    * discharging is capped by the realised SoC and the PCS;
    * export is capped by the grid/contract limit of that hour;
    * the SoC, throughput and cycle counter are advanced by the settled flow, so
      the next window starts from the true physical state.

    In the perfect-foresight benchmark (forecast == realised) this reproduces
    the planned dispatch exactly, which keeps the benchmark comparable.

    Returns (new_ending_soc, new_cumulative_throughput). `BESS_TotalCycles` is
    reported as a *cumulative* quantity, exactly like `simulate_single_site`.
    """
    eff_cap = float(p.bess_energy_mwh) * float(cap_factor)
    eta_ch = float(p.eta_charge)
    eta_dis = float(p.eta_discharge)
    pcs = max(float(p.pcs_power_mw), 0.0)

    n = commit
    planned_charge = np.maximum(sol["charge_in"][:n], 0.0)
    planned_import = np.maximum(
        sol.get("import", np.zeros(len(sol["charge_in"])))[:n], 0.0
    )
    planned_discharge = np.maximum(sol["discharge"][:n], 0.0)
    # Planned plant export. In causal mode it is deliberately NOT trusted: the
    # realised generation decides what can be exported. In oracle mode the plan
    # was made on the realised series, so it is honoured to keep the benchmark
    # an exact replay of the optimiser's own solution.
    planned_gen_export = np.maximum(sol["gen_to_grid"][:n], 0.0)

    solar = df["solar_MWh"].to_numpy(dtype=float)[start : start + n]
    wind = df["wind_MWh"].to_numpy(dtype=float)[start : start + n]
    gen = np.maximum(solar + wind, 0.0)
    peak = df["is_peak"].to_numpy(dtype=int)[start : start + n]

    max_energy = eff_cap * (p.max_soc_pct / 100.0)
    min_energy = eff_cap * (p.min_soc_pct / 100.0)
    grid_cap = None if p.grid_limit_mw is None else max(float(p.grid_limit_mw), 0.0)

    if eff_cap > 0.0:
        soc = float(np.clip(soc_before, min_energy, max_energy))
    else:
        soc = 0.0

    charge_in = np.zeros(n, dtype=float)
    plant_charge_arr = np.zeros(n, dtype=float)
    discharge = np.zeros(n, dtype=float)
    gen_to_grid = np.zeros(n, dtype=float)
    curtailed = np.zeros(n, dtype=float)
    penalty = np.zeros(n, dtype=float)
    soc_out = np.zeros(n, dtype=float)
    total_supplied = np.zeros(n, dtype=float)

    for t in range(n):
        gen_t = float(gen[t])
        if int(peak[t]) == 1:
            # ---- PEAK: honour planned charging, then BESS fills the shortfall ----
            # `no_charge_in_peak` normally zeroes peak charging inside the
            # optimiser. When the user explicitly allows it, the settlement must
            # execute the plan instead of silently overriding it, otherwise the
            # oracle benchmark would not reproduce the optimiser's own plan.
            target = (
                max(float(p.contract_mw), 0.0)
                if grid_cap is None
                else min(max(float(p.contract_mw), 0.0), grid_cap)
            )
            room = max(max_energy - soc, 0.0) if eff_cap > 0 else 0.0
            planned_plant_charge = max(planned_charge[t] - planned_import[t], 0.0)
            charge = min(planned_plant_charge, gen_t, pcs, room / max(eta_ch, 1e-9))
            charge_plant = charge
            imp = min(
                planned_import[t],
                max(pcs - charge, 0.0),
                max(room - charge * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            charge += imp
            soc_after_charge = (
                min(soc + charge * eta_ch, max_energy) if eff_cap > 0 else 0.0
            )
            gen_avail = max(gen_t - charge_plant, 0.0)
            if oracle:
                g2g = min(float(planned_gen_export[t]), gen_avail, target)
            else:
                g2g = min(gen_avail, target)
            shortfall = max(target - g2g, 0.0)
            avail = (
                max((soc_after_charge - min_energy) * eta_dis, 0.0)
                if eff_cap > 0
                else 0.0
            )
            # The plan decides whether the battery is used at all, the physics
            # decides how much is possible.
            dis = min(planned_discharge[t], shortfall, avail, pcs)
            soc = (
                max(soc_after_charge - dis / eta_dis, min_energy)
                if eff_cap > 0
                else 0.0
            )
            curt = max(gen_avail - g2g, 0.0)
            pen = max(float(p.contract_mw) - (g2g + dis), 0.0)
        else:
            # ---- OFF-PEAK: charge from own generation first, then export ----
            room = max(max_energy - soc, 0.0) if eff_cap > 0 else 0.0
            planned_plant_charge = max(planned_charge[t] - planned_import[t], 0.0)
            charge = min(planned_plant_charge, gen_t, pcs, room / max(eta_ch, 1e-9))
            charge_plant = charge
            imp = min(
                planned_import[t],
                max(pcs - charge, 0.0),
                max(room - charge * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            charge += imp
            soc_after_charge = min(
                soc + charge * eta_ch, max_energy
            ) if eff_cap > 0 else 0.0
            avail = (
                max((soc_after_charge - min_energy) * eta_dis, 0.0)
                if eff_cap > 0
                else 0.0
            )
            dis = min(planned_discharge[t], avail, pcs)
            soc = max(soc_after_charge - dis / eta_dis, min_energy) if eff_cap > 0 else 0.0

            gen_avail = max(gen_t - (charge - imp), 0.0)
            if oracle:
                g2g = min(float(planned_gen_export[t]), gen_avail)
                if grid_cap is not None:
                    g2g = min(g2g, max(grid_cap - dis, 0.0))
            elif grid_cap is None:
                g2g = gen_avail
            else:
                g2g = min(gen_avail, max(grid_cap - dis, 0.0))
            curt = max(gen_avail - g2g, 0.0)
            pen = 0.0

        charge_in[t] = charge
        plant_charge_arr[t] = charge_plant
        discharge[t] = dis
        gen_to_grid[t] = g2g
        curtailed[t] = curt
        penalty[t] = pen
        soc_out[t] = soc
        total_supplied[t] = g2g + dis

    charge_to_soc = charge_in * eta_ch
    charge_loss = np.maximum(charge_in - charge_to_soc, 0.0)

    # Reporting split between solar and wind, proportional to generation
    # (identical convention to `simulate_single_site`).
    with np.errstate(divide="ignore", invalid="ignore"):
        solar_frac = np.where(gen > 0.0, solar / np.where(gen > 0.0, gen, 1.0), 0.0)
    solar_frac = np.nan_to_num(solar_frac)

    # Split the plant charge between solar and wind before splitting the export.
    plant_charge = np.minimum(plant_charge_arr, np.maximum(gen - gen_to_grid, 0.0))
    s_charge = plant_charge * solar_frac
    w_charge = plant_charge - s_charge

    s_pot = np.maximum(solar - s_charge, 0.0)
    w_pot = np.maximum(wind - w_charge, 0.0)
    pot = s_pot + w_pot
    with np.errstate(divide="ignore", invalid="ignore"):
        s_share = np.where(pot > 0.0, s_pot / np.where(pot > 0.0, pot, 1.0), 0.0)
    s_share = np.nan_to_num(s_share)

    s_to_grid = gen_to_grid * s_share
    w_to_grid = gen_to_grid - s_to_grid

    iloc_idx = np.arange(start, start + n)
    df.loc[iloc_idx, "BESS_SOC_MWh"] = soc_out
    df.loc[iloc_idx, "BESS_SOC_pct"] = (
        100.0 * soc_out / eff_cap if eff_cap > 0 else 0.0
    )
    df.loc[iloc_idx, "BESS_Charge_in_MWh"] = charge_in
    df.loc[iloc_idx, "BESS_Charge_to_SOC_MWh"] = charge_to_soc
    df.loc[iloc_idx, "BESS_Charge_Loss_MWh"] = charge_loss
    df.loc[iloc_idx, "BESS_Discharge_MWh"] = discharge
    df.loc[iloc_idx, "solar_to_grid_MWh"] = s_to_grid
    df.loc[iloc_idx, "wind_to_grid_MWh"] = w_to_grid
    df.loc[iloc_idx, "GenPlant_to_Grid_MWh"] = gen_to_grid
    df.loc[iloc_idx, "BESS_to_Grid_MWh"] = discharge
    df.loc[iloc_idx, "Total_Supplied_MWh"] = total_supplied
    df.loc[iloc_idx, "Curtailed_MWh"] = curtailed
    df.loc[iloc_idx, "Peak_Time_Energy_Supplied_MWh"] = np.where(peak == 1, total_supplied, 0.0)
    df.loc[iloc_idx, "Offpeak_Supplied_MWh"] = np.where(peak == 0, total_supplied, 0.0)
    df.loc[iloc_idx, "Penalty_MWh"] = penalty

    cum_throughput = float(cum_throughput_before) + float(
        np.sum(charge_in) + np.sum(discharge) / max(eta_dis, 1e-9)
    )
    df.loc[iloc_idx, "BESS_TotalCycles"] = (
        cum_throughput / (2.0 * eff_cap) if eff_cap > 0 else 0.0
    )

    # Return the settled SoC at the end of the committed block.
    return float(soc_out[-1]), cum_throughput


# -----------------------------
# Optimised split-site driver
# -----------------------------
def simulate_optimized_split_site(
    df_in: pd.DataFrame,
    p: BESSParams,
    o: OptimizerParams,
    cf_solar: float,
    cf_wind: float,
    initial_soc_solar_mwh: Optional[float] = None,
    initial_soc_wind_mwh: Optional[float] = None,
    progress_bar=None,
    forecast_history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, float, float, Dict[str, Any]]:
    """
    Optimised dispatch, split site (solar BESS + wind BESS, each fed only by its
    own source). Output schema matches `simulate_split_site`.

    FLAW 9 FIX - same true receding-horizon contract as the single-site driver:
    solve on a causal forecast, settle the committed hours against realised
    generation, and carry the physical SoC/throughput to the next step.
    """
    solver, solver_label = resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_binaries, bin_reason = needs_complementarity(p, o)

    df = df_in.copy().reset_index(drop=True)
    df = ensure_hour_int(df)
    df["solar_MWh"] = df["solar_MWh"].astype(float)
    df["wind_MWh"] = df["wind_MWh"].astype(float)
    df["gen_MWh"] = df["solar_MWh"] + df["wind_MWh"]
    df["is_peak"] = df["Hour"].isin(p.peak_hours).astype(int)

    hours_all = df["Hour"].to_numpy(dtype=int)
    prior_solar = prior_wind = prior_hours = None
    if forecast_history is not None and len(forecast_history):
        hist = ensure_hour_int(forecast_history.copy())
        prior_solar = hist["solar_MWh"].to_numpy(dtype=float)
        prior_wind = hist["wind_MWh"].to_numpy(dtype=float)
        prior_hours = hist["Hour"].to_numpy(dtype=int)

    solar_cap_y = float(p.solar_bess_energy_mwh) * float(cf_solar)
    wind_cap_y = float(p.wind_bess_energy_mwh) * float(cf_wind)

    df["solar_bess_capacity_MWh"] = solar_cap_y
    df["wind_bess_capacity_MWh"] = wind_cap_y

    for col in [
        "solar_bess_soc_MWh",
        "solar_bess_soc_pct",
        "solar_bess_charge_in_MWh",
        "solar_bess_charge_to_soc_MWh",
        "solar_bess_loss_MWh",
        "solar_bess_discharge_MWh",
        "solar_bess_total_cycles",
        "wind_bess_soc_MWh",
        "wind_bess_soc_pct",
        "wind_bess_charge_in_MWh",
        "wind_bess_charge_to_soc_MWh",
        "wind_bess_loss_MWh",
        "wind_bess_discharge_MWh",
        "wind_bess_total_cycles",
        "solar_to_grid_MWh",
        "solar_to_bess_MWh",
        "wind_to_grid_MWh",
        "wind_to_bess_MWh",
        "GenPlant_to_Grid_MWh",
        "BESS_to_Grid_MWh",
        "Total_Supplied_MWh",
        "Curtailed_MWh",
        "Peak_Time_Energy_Supplied_MWh",
        "Offpeak_Supplied_MWh",
        "Penalty_MWh",
    ]:
        df[col] = 0.0

    s_max = solar_cap_y * (p.max_soc_pct / 100.0)
    s_min = solar_cap_y * (p.min_soc_pct / 100.0)
    w_max = wind_cap_y * (p.max_soc_pct / 100.0)
    w_min = wind_cap_y * (p.min_soc_pct / 100.0)

    solar_soc = (
        s_min + (float(p.initial_soc_pct) / 100.0) * (s_max - s_min)
        if initial_soc_solar_mwh is None
        else float(np.clip(initial_soc_solar_mwh, min(s_min, s_max), max(s_min, s_max)))
    )
    wind_soc = (
        w_min + (float(p.initial_soc_pct) / 100.0) * (w_max - w_min)
        if initial_soc_wind_mwh is None
        else float(np.clip(initial_soc_wind_mwh, min(w_min, w_max), max(w_min, w_max)))
    )

    eta_dis = float(p.eta_discharge)
    cum_thr_s = 0.0
    cum_thr_w = 0.0

    n = len(df)
    windows = _year_windows(n, o.horizon_hours, o.step_hours)
    diag = {
        "solver": solver_label,
        "formulation": "MILP (complementarity binaries)" if use_binaries else "LP (exact relaxation)",
        "complementarity_reason": bin_reason,
        "windows": 0,
        "fallback_windows": 0,
        "degenerate_windows": 0,
        "milp_rescues": 0,
        "solve_seconds": 0.0,
        "objective_first_window": None,
        "last_error": None,
    }

    for wi, (start, stop, commit) in enumerate(windows):
        # FLAW 9 FIX - optimiser sees only the causal forecast.
        win = _forecast_window(
            df, start, stop, hours_all, o.forecast_mode, o.forecast_method,
            prior_solar, prior_wind, prior_hours,
        )

        t0 = time.perf_counter()
        try:
            sol = _optimize_split_window(
                win, p, o, solar_cap_y, wind_cap_y, solar_soc, wind_soc, solver, use_binaries
            )
            if sol is not None and not use_binaries:
                sc = _max_simultaneous(sol["cs"] + sol["imp"], sol["ds"] + sol["dw"])
                if sc > SIMULTANEITY_TOL_MWH:
                    diag["degenerate_windows"] += 1
                    forced = replace(o, force_complementarity=True)
                    sol_milp = _optimize_split_window(
                        win, p, forced, solar_cap_y, wind_cap_y,
                        solar_soc, wind_soc, solver, True,
                    )
                    if sol_milp is not None:
                        sol = sol_milp
                        diag["milp_rescues"] += 1
        except Exception as exc:
            sol = None
            diag["last_error"] = f"{type(exc).__name__}: {exc}"
        diag["solve_seconds"] += time.perf_counter() - t0
        diag["windows"] += 1

        if sol is None:
            # FLAW 9 FIX - fall back on the committed rows only.
            diag["fallback_windows"] += 1
            committed = df.iloc[start : start + commit].copy()
            block, solar_soc, wind_soc = simulate_split_site(
                committed,
                p,
                cf_solar,
                cf_wind,
                initial_soc_solar_mwh=solar_soc,
                initial_soc_wind_mwh=wind_soc,
            )
            block = block.reset_index(drop=True)
            # Per-row cumulative cycle counters, re-based on the global state.
            blk_cum_s = (
                block["solar_bess_charge_in_MWh"].cumsum()
                + block["solar_bess_discharge_MWh"].cumsum() / max(eta_dis, 1e-9)
            )
            blk_cum_w = (
                block["wind_bess_charge_in_MWh"].cumsum()
                + block["wind_bess_discharge_MWh"].cumsum() / max(eta_dis, 1e-9)
            )
            if solar_cap_y > 0.0:
                block["solar_bess_total_cycles"] = (
                    cum_thr_s + blk_cum_s
                ) / (2.0 * solar_cap_y)
            if wind_cap_y > 0.0:
                block["wind_bess_total_cycles"] = (
                    cum_thr_w + blk_cum_w
                ) / (2.0 * wind_cap_y)
            cum_thr_s += float(blk_cum_s.iloc[-1]) if len(blk_cum_s) else 0.0
            cum_thr_w += float(blk_cum_w.iloc[-1]) if len(blk_cum_w) else 0.0
            for col in df.columns:
                if col in block.columns:
                    df.iloc[
                        start : start + commit, df.columns.get_loc(col)
                    ] = block[col].to_numpy()
        else:
            solar_soc, wind_soc, cum_thr_s, cum_thr_w = _write_split_block(
                df, start, commit, sol, p, solar_cap_y, wind_cap_y,
                solar_soc, wind_soc, cum_thr_s, cum_thr_w,
                oracle=str(o.forecast_mode).lower() != "causal",
            )

        if wi % 24 == 0 or wi == len(windows) - 1:
            _update_bar(
                progress_bar, wi + 1, len(windows),
                f"Optimising year: window {wi + 1}/{len(windows)}",
            )

    return df, solar_soc, wind_soc, diag


def _optimize_split_window(
    win: pd.DataFrame,
    p: BESSParams,
    o: OptimizerParams,
    solar_cap_y: float,
    wind_cap_y: float,
    solar_soc_start: float,
    wind_soc_start: float,
    solver,
    use_binaries: bool,
) -> Optional[Dict[str, Any]]:
    """MILP/LP for one split-site window."""
    T = len(win)
    solar = win["solar_MWh"].to_numpy(dtype=float)
    wind = win["wind_MWh"].to_numpy(dtype=float)
    total_gen = solar + wind
    peak = win["is_peak"].to_numpy(dtype=int)
    hours = win["Hour"].to_numpy()

    # Degeneracy breaker for the LP relaxation (see OptimizerParams docstring).
    w_throughput = float(o.w_degradation)
    if not use_binaries and w_throughput <= 0.0:
        w_throughput = max(float(o.degeneracy_epsilon), 0.0)

    s_max = solar_cap_y * (p.max_soc_pct / 100.0)
    s_min = solar_cap_y * (p.min_soc_pct / 100.0)
    w_max = wind_cap_y * (p.max_soc_pct / 100.0)
    w_min = wind_cap_y * (p.min_soc_pct / 100.0)

    # FLAW 6 FIX - split the contract between the two technologies. The same
    # effective-contract rule as the rule-based split engine is used, so both
    # engines judge each unit against the same target.
    if p.grid_limit_mw is None:
        _eff_contract = max(float(p.contract_mw), 0.0)
    else:
        _eff_contract = min(
            max(float(p.contract_mw), 0.0), max(float(p.grid_limit_mw), 0.0)
        )
    _share = float(np.clip(p.solar_contract_share, 0.0, 1.0))
    solar_target = _eff_contract * _share
    wind_target = _eff_contract * (1.0 - _share)

    e_s = [plp.LpVariable(f"es_{t}", lowBound=s_min, upBound=s_max) for t in range(T + 1)]
    e_w = [plp.LpVariable(f"ew_{t}", lowBound=w_min, upBound=w_max) for t in range(T + 1)]

    p_solar = max(float(p.solar_pcs_power_mw), 0.0)
    p_wind = max(float(p.wind_pcs_power_mw), 0.0)

    cs = [plp.LpVariable(f"cs_{t}", lowBound=0.0, upBound=p_solar) for t in range(T)]
    ds = [plp.LpVariable(f"ds_{t}", lowBound=0.0, upBound=p_solar) for t in range(T)]
    cw = [plp.LpVariable(f"cw_{t}", lowBound=0.0, upBound=p_wind) for t in range(T)]
    dw = [plp.LpVariable(f"dw_{t}", lowBound=0.0, upBound=p_wind) for t in range(T)]
    gs = [plp.LpVariable(f"gs_{t}", lowBound=0.0, upBound=float(solar[t])) for t in range(T)]
    gw = [plp.LpVariable(f"gw_{t}", lowBound=0.0, upBound=float(wind[t])) for t in range(T)]
    imp = [plp.LpVariable(f"imp_{t}", lowBound=0.0, upBound=p_solar + p_wind) for t in range(T)]
    # FLAW 6 FIX - one shortfall variable per technology, each against its own
    # share of the contract, in peak hours only.
    pen_s = [
        plp.LpVariable(
            f"pens_{t}", lowBound=0.0,
            upBound=solar_target if peak[t] == 1 else 0.0,
        )
        for t in range(T)
    ]
    pen_w = [
        plp.LpVariable(
            f"penw_{t}", lowBound=0.0,
            upBound=wind_target if peak[t] == 1 else 0.0,
        )
        for t in range(T)
    ]
    cur_s = [plp.LpVariable(f"curs_{t}", lowBound=0.0, upBound=float(solar[t])) for t in range(T)]
    cur_w = [plp.LpVariable(f"curw_{t}", lowBound=0.0, upBound=float(wind[t])) for t in range(T)]

    u = [plp.LpVariable(f"u_{t}", cat="Binary") for t in range(T)] if use_binaries else None

    eta_ch = float(p.eta_charge)
    eta_dis = float(p.eta_discharge)
    inv_eta_dis = 1.0 / max(eta_dis, 1e-9)
    rho = 1.0 - max(0.0, min(0.999, float(o.self_discharge_per_hour)))

    big_m = float(np.sum(total_gen)) + p_solar + p_wind + max(abs(p.contract_mw), 0.0) + 1.0
    caps = _peak_export_caps(peak, total_gen, p.contract_mw, p.grid_limit_mw, big_m)
    tariff = build_tariff_vector(hours, o.tariff, o.spike_filter_window, o.spike_filter_k)

    m = plp.LpProblem("BESS_Dispatch_Split", plp.LpMinimize)

    m += e_s[0] == float(np.clip(solar_soc_start, min(s_min, s_max), max(s_min, s_max)))
    m += e_w[0] == float(np.clip(wind_soc_start, min(w_min, w_max), max(w_min, w_max)))

    for t in range(T):
        # Grid exchange cap (identical semantics to the single-mode caps).
        m += gs[t] + gw[t] + ds[t] + dw[t] - imp[t] <= caps[t] + 1e-7, f"export_cap_{t}"

        # Each BESS may only absorb its own source.
        m += gs[t] + cs[t] <= solar[t] + 1e-7, f"solar_balance_{t}"
        m += gw[t] + cw[t] <= wind[t] + 1e-7, f"wind_balance_{t}"

        if not o.allow_grid_charging:
            m += imp[t] <= 1e-7, f"no_grid_import_{t}"
        if o.no_charge_in_peak and peak[t] == 1:
            m += cs[t] + cw[t] + imp[t] <= 1e-7, f"no_peak_charge_{t}"

        # Grid import can only serve the battery, split between the two PCS.
        m += imp[t] <= p_solar + p_wind + 1e-7, f"pcs_charge_{t}"

        m += (
            e_s[t + 1] == rho * e_s[t] + eta_ch * cs[t] - inv_eta_dis * ds[t]
        ), f"soc_s_{t}"
        m += (
            e_w[t + 1] == rho * e_w[t] + eta_ch * cw[t] - inv_eta_dis * dw[t]
        ), f"soc_w_{t}"

        if u is not None:
            m += cs[t] + imp[t] <= p_solar * u[t] + 1e-7, f"comp_chs_{t}"
            m += cw[t] + imp[t] <= p_wind * u[t] + 1e-7, f"comp_chw_{t}"
            m += ds[t] + dw[t] <= (p_solar + p_wind) * (1 - u[t]) + 1e-7, f"comp_dis_{t}"

        m += cur_s[t] >= solar[t] - (gs[t] + cs[t]), f"curs_def_{t}"
        m += cur_w[t] >= wind[t] - (gw[t] + cw[t]), f"curw_def_{t}"

        # Shortfall is only defined during peak hours, and each technology is
        # measured against its own share of the contract (FLAW 6 FIX). The
        # per-technology caps mirror the rule-based engine: a plant cannot use
        # the other plant's contract headroom.
        if peak[t] == 1:
            m += gs[t] + ds[t] <= solar_target + 1e-7, f"split_peak_s_{t}"
            m += gw[t] + dw[t] <= wind_target + 1e-7, f"split_peak_w_{t}"
            m += pen_s[t] >= solar_target - (gs[t] + ds[t]), f"pen_s_def_{t}"
            m += pen_w[t] >= wind_target - (gw[t] + dw[t]), f"pen_w_def_{t}"

    if o.terminal_soc_pct is not None:
        m += e_s[T] >= solar_cap_y * (float(o.terminal_soc_pct) / 100.0) - 1e-7
        m += e_w[T] >= wind_cap_y * (float(o.terminal_soc_pct) / 100.0) - 1e-7

    if o.max_equivalent_cycles_per_year is not None:
        step = max(min(o.step_hours, T), 1)
        share = step / HOURS_IN_YEAR
        thr_s = plp.LpVariable("throughput_s", lowBound=0.0)
        thr_w = plp.LpVariable("throughput_w", lowBound=0.0)
        m += thr_s == plp.lpSum(cs[t] + inv_eta_dis * ds[t] for t in range(T))
        m += thr_w == plp.lpSum(cw[t] + inv_eta_dis * dw[t] for t in range(T))
        if solar_cap_y > 0.0:
            m += thr_s <= 2.0 * solar_cap_y * float(o.max_equivalent_cycles_per_year) * share + 1e-7
        if wind_cap_y > 0.0:
            m += thr_w <= 2.0 * wind_cap_y * float(o.max_equivalent_cycles_per_year) * share + 1e-7

    # FIX 5a - terminal SoC cost, split configuration. One deviation pair
    # per battery, because each unit has its own capacity and its own target.
    term_cost = 0.0
    if float(o.w_terminal_soc) > 0.0:
        # same rule as the single-site objective: fall back to the band
        # midpoint so the cost is never inert
        _ts_target = (
            float(o.terminal_soc_pct) / 100.0
            if o.terminal_soc_pct is not None
            else 0.5 * (float(p.min_soc_pct) + float(p.max_soc_pct)) / 100.0
        )
        _pairs = (
            ("s", e_s, solar_cap_y * _ts_target),
            ("w", e_w, wind_cap_y * _ts_target),
        )
        for _tag, _state, _tgt in _pairs:
            _dp = plp.LpVariable(f"term_dev_pos_{_tag}", lowBound=0.0)
            _dn = plp.LpVariable(f"term_dev_neg_{_tag}", lowBound=0.0)
            m += _dp >= _state[T] - _tgt, f"term_soc_above_{_tag}"
            m += _dn >= _tgt - _state[T], f"term_soc_below_{_tag}"
            term_cost += float(o.w_terminal_soc) * (_dp + _dn)

    m += plp.lpSum(
        o.w_penalty * (pen_s[t] + pen_w[t])
        + o.w_curtail * (cur_s[t] + cur_w[t])
        + o.grid_import_price * imp[t]
        + w_throughput * (cs[t] + cw[t] + inv_eta_dis * ds[t] + inv_eta_dis * dw[t])
        - tariff[t] * (gs[t] + gw[t] + ds[t] + dw[t] - imp[t])
        for t in range(T)
    ) + term_cost

    status = m.solve(solver)
    if plp.LpStatus.get(status, str(status)) not in ("Optimal", "Feasible"):
        return None

    def arr(vars_):
        return np.array(
            [0.0 if v.varValue is None else float(v.varValue) for v in vars_],
            dtype=float,
        )

    return {
        "cs": np.maximum(arr(cs), 0.0),
        "ds": np.maximum(arr(ds), 0.0),
        "cw": np.maximum(arr(cw), 0.0),
        "dw": np.maximum(arr(dw), 0.0),
        "gs": np.clip(arr(gs), 0.0, np.maximum(solar, 0.0)),
        "gw": np.clip(arr(gw), 0.0, np.maximum(wind, 0.0)),
        "imp": np.maximum(arr(imp), 0.0),
        "es": arr(e_s),
        "ew": arr(e_w),
        "es_end": arr(e_s)[1 : T + 1],
        "ew_end": arr(e_w)[1 : T + 1],
        "pen": np.maximum(arr(pen_s), 0.0) + np.maximum(arr(pen_w), 0.0),
        "pen_solar": np.maximum(arr(pen_s), 0.0),
        "pen_wind": np.maximum(arr(pen_w), 0.0),
        "cur": np.maximum(arr(cur_s), 0.0) + np.maximum(arr(cur_w), 0.0),
    }


def _write_split_block(
    df: pd.DataFrame,
    start: int,
    commit: int,
    sol: Dict[str, Any],
    p: BESSParams,
    solar_cap_y: float,
    wind_cap_y: float,
    solar_soc_before: float,
    wind_soc_before: float,
    cum_thr_s: float = 0.0,
    cum_thr_w: float = 0.0,
    oracle: bool = False,
) -> Tuple[float, float, float, float]:
    """Settle the committed part of a split window against realised generation.

    FLAW 9 FIX - same physical replay as `_write_single_block`, applied to the
    two independent batteries. Each battery can only charge from its own
    source, and each is penalised against its own contract share in peak hours.

    Returns (solar_soc, wind_soc, cum_throughput_solar, cum_throughput_wind).
    """
    n = commit
    sl = slice(0, n)

    plan_cs = np.maximum(sol["cs"][sl], 0.0)
    plan_ds = np.maximum(sol["ds"][sl], 0.0)
    plan_cw = np.maximum(sol["cw"][sl], 0.0)
    plan_dw = np.maximum(sol["dw"][sl], 0.0)
    plan_imp = np.maximum(sol["imp"][sl], 0.0)
    # Planned plant exports, honoured only in oracle mode (see _write_single_block).
    plan_gs = np.maximum(sol["gs"][sl], 0.0)
    plan_gw = np.maximum(sol["gw"][sl], 0.0)

    solar = np.maximum(df["solar_MWh"].to_numpy(dtype=float)[start : start + n], 0.0)
    wind = np.maximum(df["wind_MWh"].to_numpy(dtype=float)[start : start + n], 0.0)
    peak = df["is_peak"].to_numpy(dtype=int)[start : start + n]

    eta_ch = float(p.eta_charge)
    eta_dis = float(p.eta_discharge)
    pcs_s = max(float(p.solar_pcs_power_mw), 0.0)
    pcs_w = max(float(p.wind_pcs_power_mw), 0.0)

    s_max = solar_cap_y * (p.max_soc_pct / 100.0)
    s_min = solar_cap_y * (p.min_soc_pct / 100.0)
    w_max = wind_cap_y * (p.max_soc_pct / 100.0)
    w_min = wind_cap_y * (p.min_soc_pct / 100.0)
    grid_cap = None if p.grid_limit_mw is None else max(float(p.grid_limit_mw), 0.0)

    solar_soc = (
        float(np.clip(solar_soc_before, s_min, s_max)) if solar_cap_y > 0 else 0.0
    )
    wind_soc = (
        float(np.clip(wind_soc_before, w_min, w_max)) if wind_cap_y > 0 else 0.0
    )

    # FLAW 6: the contract is split between the technologies, so settlement must
    # judge each against its own target.
    if grid_cap is None:
        _eff_contract = max(float(p.contract_mw), 0.0)
    else:
        _eff_contract = min(max(float(p.contract_mw), 0.0), grid_cap)
    _share = float(np.clip(p.solar_contract_share, 0.0, 1.0))
    s_target = _eff_contract * _share
    w_target = _eff_contract * (1.0 - _share)

    cs = np.zeros(n, dtype=float)
    cw = np.zeros(n, dtype=float)
    cs_plant = np.zeros(n, dtype=float)
    cw_plant = np.zeros(n, dtype=float)
    ds = np.zeros(n, dtype=float)
    dw = np.zeros(n, dtype=float)
    gs = np.zeros(n, dtype=float)
    gw = np.zeros(n, dtype=float)
    curtailed = np.zeros(n, dtype=float)
    penalty = np.zeros(n, dtype=float)
    es_out = np.zeros(n, dtype=float)
    ew_out = np.zeros(n, dtype=float)
    total_supplied = np.zeros(n, dtype=float)

    # Grid import is attributed to the solar PCS first (same convention as the
    # optimiser's own settlement and the existing reporting split).
    imp_s = np.minimum(plan_imp, plan_cs) if n else np.zeros(0)
    imp_w = plan_imp - imp_s

    for t in range(n):
        solar_t = float(solar[t])
        wind_t = float(wind[t])

        if int(peak[t]) == 1:
            # ---- PEAK: honour planned charging, then each BESS fills own gap ----
            room_s = max(s_max - solar_soc, 0.0) if solar_cap_y > 0 else 0.0
            c_s = min(plan_cs[t], solar_t, pcs_s, room_s / max(eta_ch, 1e-9))
            cs_plant_t = c_s
            imp_s_t = min(
                imp_s[t],
                max(pcs_s - c_s, 0.0),
                max(room_s - c_s * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            c_s += imp_s_t
            soc_s_after = (
                min(solar_soc + c_s * eta_ch, s_max) if solar_cap_y > 0 else 0.0
            )
            s_gen_avail = max(solar_t - cs_plant_t, 0.0)

            room_w = max(w_max - wind_soc, 0.0) if wind_cap_y > 0 else 0.0
            c_w = min(plan_cw[t], wind_t, pcs_w, room_w / max(eta_ch, 1e-9))
            cw_plant_t = c_w
            imp_w_t = min(
                imp_w[t],
                max(pcs_w - c_w, 0.0),
                max(room_w - c_w * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            c_w += imp_w_t
            soc_w_after = (
                min(wind_soc + c_w * eta_ch, w_max) if wind_cap_y > 0 else 0.0
            )
            w_gen_avail = max(wind_t - cw_plant_t, 0.0)

            s2g = min(s_gen_avail, s_target)
            w2g = min(w_gen_avail, w_target)
            if oracle:
                s2g = min(float(plan_gs[t]), s_gen_avail, s_target)
                w2g = min(float(plan_gw[t]), w_gen_avail, w_target)
            s_short = max(s_target - s2g, 0.0)
            w_short = max(w_target - w2g, 0.0)

            avail_s = (
                max((soc_s_after - s_min) * eta_dis, 0.0) if solar_cap_y > 0 else 0.0
            )
            avail_w = (
                max((soc_w_after - w_min) * eta_dis, 0.0) if wind_cap_y > 0 else 0.0
            )
            d_s = min(plan_ds[t], s_short, avail_s, pcs_s)
            d_w = min(plan_dw[t], w_short, avail_w, pcs_w)
            solar_soc = (
                max(soc_s_after - d_s / eta_dis, s_min) if solar_cap_y > 0 else 0.0
            )
            wind_soc = (
                max(soc_w_after - d_w / eta_dis, w_min) if wind_cap_y > 0 else 0.0
            )

            curt = max(s_gen_avail - s2g, 0.0) + max(w_gen_avail - w2g, 0.0)
            pen = max(s_target - (s2g + d_s), 0.0) + max(
                w_target - (w2g + d_w), 0.0
            )
        else:
            # ---- OFF-PEAK: each battery charges from its own source ----
            room_s = max(s_max - solar_soc, 0.0) if solar_cap_y > 0 else 0.0
            c_s = min(plan_cs[t], solar_t, pcs_s, room_s / max(eta_ch, 1e-9))
            cs_plant_t = c_s
            imp_s_t = min(
                imp_s[t],
                max(pcs_s - c_s, 0.0),
                max(room_s - c_s * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            c_s += imp_s_t
            soc_s_after = (
                min(solar_soc + c_s * eta_ch, s_max) if solar_cap_y > 0 else 0.0
            )

            room_w = max(w_max - wind_soc, 0.0) if wind_cap_y > 0 else 0.0
            c_w = min(plan_cw[t], wind_t, pcs_w, room_w / max(eta_ch, 1e-9))
            cw_plant_t = c_w
            imp_w_t = min(
                imp_w[t],
                max(pcs_w - c_w, 0.0),
                max(room_w - c_w * eta_ch, 0.0) / max(eta_ch, 1e-9),
            )
            c_w += imp_w_t
            soc_w_after = (
                min(wind_soc + c_w * eta_ch, w_max) if wind_cap_y > 0 else 0.0
            )

            avail_s = (
                max((soc_s_after - s_min) * eta_dis, 0.0) if solar_cap_y > 0 else 0.0
            )
            avail_w = (
                max((soc_w_after - w_min) * eta_dis, 0.0) if wind_cap_y > 0 else 0.0
            )
            d_s = min(plan_ds[t], avail_s, pcs_s)
            d_w = min(plan_dw[t], avail_w, pcs_w)
            solar_soc = (
                max(soc_s_after - d_s / eta_dis, s_min) if solar_cap_y > 0 else 0.0
            )
            wind_soc = (
                max(soc_w_after - d_w / eta_dis, w_min) if wind_cap_y > 0 else 0.0
            )

            # Combined export cap applies to the two plants' own generation.
            s_gen_avail = max(solar_t - c_s + imp_s_t, 0.0)
            w_gen_avail = max(wind_t - c_w + imp_w_t, 0.0)
            total_gen_avail = s_gen_avail + w_gen_avail
            if oracle:
                s2g = min(float(plan_gs[t]), s_gen_avail)
                w2g = min(float(plan_gw[t]), w_gen_avail)
                if grid_cap is not None:
                    total_export_room = max(grid_cap - (d_s + d_w), 0.0)
                    total_planned = s2g + w2g
                    if total_planned > total_export_room and total_planned > 0.0:
                        frac = total_export_room / total_planned
                        s2g *= frac
                        w2g *= frac
            elif grid_cap is None:
                s2g, w2g = s_gen_avail, w_gen_avail
            else:
                total_export_room = max(grid_cap - (d_s + d_w), 0.0)
                if total_gen_avail > 0.0:
                    frac = min(total_export_room / total_gen_avail, 1.0)
                else:
                    frac = 0.0
                s2g = s_gen_avail * frac
                w2g = w_gen_avail * frac
            curt = (s_gen_avail - s2g) + (w_gen_avail - w2g)
            pen = 0.0

        cs[t], cw[t] = c_s, c_w
        cs_plant[t], cw_plant[t] = cs_plant_t, cw_plant_t
        ds[t], dw[t] = d_s, d_w
        gs[t], gw[t] = s2g, w2g
        curtailed[t] = curt
        penalty[t] = pen
        es_out[t] = solar_soc
        ew_out[t] = wind_soc
        total_supplied[t] = s2g + w2g + d_s + d_w

    cs_to_soc = cs * eta_ch
    cw_to_soc = cw * eta_ch

    iloc_idx = np.arange(start, start + n)
    df.loc[iloc_idx, "solar_bess_soc_MWh"] = es_out
    df.loc[iloc_idx, "solar_bess_soc_pct"] = (
        100.0 * es_out / solar_cap_y if solar_cap_y > 0 else 0.0
    )
    df.loc[iloc_idx, "solar_bess_charge_in_MWh"] = cs
    df.loc[iloc_idx, "solar_bess_charge_to_soc_MWh"] = cs_to_soc
    df.loc[iloc_idx, "solar_bess_loss_MWh"] = np.maximum(cs - cs_to_soc, 0.0)
    df.loc[iloc_idx, "solar_bess_discharge_MWh"] = ds
    df.loc[iloc_idx, "wind_bess_soc_MWh"] = ew_out
    df.loc[iloc_idx, "wind_bess_soc_pct"] = (
        100.0 * ew_out / wind_cap_y if wind_cap_y > 0 else 0.0
    )
    df.loc[iloc_idx, "wind_bess_charge_in_MWh"] = cw
    df.loc[iloc_idx, "wind_bess_charge_to_soc_MWh"] = cw_to_soc
    df.loc[iloc_idx, "wind_bess_loss_MWh"] = np.maximum(cw - cw_to_soc, 0.0)
    df.loc[iloc_idx, "wind_bess_discharge_MWh"] = dw

    df.loc[iloc_idx, "solar_to_grid_MWh"] = gs
    df.loc[iloc_idx, "solar_to_bess_MWh"] = cs_plant
    df.loc[iloc_idx, "wind_to_grid_MWh"] = gw
    df.loc[iloc_idx, "wind_to_bess_MWh"] = cw_plant

    df.loc[iloc_idx, "GenPlant_to_Grid_MWh"] = gs + gw
    df.loc[iloc_idx, "BESS_to_Grid_MWh"] = ds + dw
    df.loc[iloc_idx, "Total_Supplied_MWh"] = total_supplied
    df.loc[iloc_idx, "Curtailed_MWh"] = curtailed
    df.loc[iloc_idx, "Peak_Time_Energy_Supplied_MWh"] = np.where(
        peak == 1, total_supplied, 0.0
    )
    df.loc[iloc_idx, "Offpeak_Supplied_MWh"] = np.where(
        peak == 0, total_supplied, 0.0
    )
    df.loc[iloc_idx, "Penalty_MWh"] = penalty

    # Same cumulative throughput / cycle convention as `simulate_split_site`.
    thr_s = cum_thr_s + float(np.sum(cs) + np.sum(ds) / max(eta_dis, 1e-9))
    thr_w = cum_thr_w + float(np.sum(cw) + np.sum(dw) / max(eta_dis, 1e-9))
    df.loc[iloc_idx, "solar_bess_total_cycles"] = (
        thr_s / (2.0 * solar_cap_y) if solar_cap_y > 0 else 0.0
    )
    df.loc[iloc_idx, "wind_bess_total_cycles"] = (
        thr_w / (2.0 * wind_cap_y) if wind_cap_y > 0 else 0.0
    )

    if n > 0:
        return float(es_out[-1]), float(ew_out[-1]), thr_s, thr_w
    return solar_soc, wind_soc, thr_s, thr_w


def simulate_optimized_multi_year(
    df_base: pd.DataFrame,
    p: BESSParams,
    o: OptimizerParams,
    degr_single: pd.Series,
    degr_solar: pd.Series,
    degr_wind: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Multi-year optimised driver with per-year degradation and optional SoC
    carry-over. Mirrors `simulate_multi_year` but calls the optimiser.
    """
    all_years: List[pd.DataFrame] = []

    shared_soc: Optional[float] = None
    solar_soc: Optional[float] = None
    wind_soc: Optional[float] = None

    years = sorted(df_base["Year"].unique())
    bar = _progressive_bar(len(years), "Preparing optimised simulation…")

    # FLAW 1 + FLAW 2 FIX - same ordinal mapping as the rule-based driver: the
    # degradation and RTE tables are keyed 1..N (project year), not by calendar
    # year. Without this, both lookups always missed.
    year_ordinal = {int(y): i + 1 for i, y in enumerate(years)}

    agg: Dict[str, Any] = {
        "solver": None,
        "formulation": None,
        "complementarity_reason": None,
        "windows": 0,
        "fallback_windows": 0,
        "degenerate_windows": 0,
        "milp_rescues": 0,
        "solve_seconds": 0.0,
        "last_error": None,
    }

    for yi, y in enumerate(years):
        df_y = df_base[df_base["Year"] == y].copy()
        year_index = year_ordinal[int(y)]

        if p.rte_by_year is not None and year_index in p.rte_by_year.index:
            eta_ch_y = float(p.rte_by_year.loc[year_index, "eta_charge"])
            eta_dis_y = float(p.rte_by_year.loc[year_index, "eta_discharge"])
        else:
            eta_ch_y = p.eta_charge
            eta_dis_y = p.eta_discharge

        p_year = BESSParams(
            solar_cap_mw=p.solar_cap_mw,
            wind_cap_mw=p.wind_cap_mw,
            contract_mw=p.contract_mw,
            bess_energy_mwh=p.bess_energy_mwh,
            pcs_power_mw=p.pcs_power_mw,
            solar_bess_energy_mwh=p.solar_bess_energy_mwh,
            solar_pcs_power_mw=p.solar_pcs_power_mw,
            wind_bess_energy_mwh=p.wind_bess_energy_mwh,
            wind_pcs_power_mw=p.wind_pcs_power_mw,
            eta_charge=eta_ch_y,
            eta_discharge=eta_dis_y,
            min_soc_pct=p.min_soc_pct,
            max_soc_pct=p.max_soc_pct,
            initial_soc_pct=p.initial_soc_pct,
            peak_hours=p.peak_hours,
            grid_limit_mw=p.grid_limit_mw,
            mode=p.mode,
            solar_contract_share=p.solar_contract_share,
            n_years=p.n_years,
            rte_by_year=p.rte_by_year,
            reset_battery_each_year=p.reset_battery_each_year,
        )

        # FLAW 9 FIX - the first hours of a new simulated year need a previous
        # day for the seasonal-naive forecast. Pass the tail of the years
        # already simulated so the causal forecast survives the year boundary.
        history_tail = (
            pd.concat(all_years, ignore_index=True).tail(48) if all_years else None
        )

        if p.mode == "single":
            cf = float(degr_single.get(year_index, 1.0))
            init_soc = None if (p.reset_battery_each_year or shared_soc is None) else shared_soc

            hourly_y, shared_soc, diag = simulate_optimized_single_site(
                df_y, p_year, o, cf, initial_soc_mwh=init_soc,
                forecast_history=history_tail,
            )
        else:
            cf_s = float(degr_solar.get(year_index, 1.0))
            cf_w = float(degr_wind.get(year_index, 1.0))
            init_s = None if (p.reset_battery_each_year or solar_soc is None) else solar_soc
            init_w = None if (p.reset_battery_each_year or wind_soc is None) else wind_soc

            hourly_y, solar_soc, wind_soc, diag = simulate_optimized_split_site(
                df_y, p_year, o, cf_s, cf_w, init_s, init_w,
                forecast_history=history_tail,
            )

        # FLAW 8 FIX - counters must accumulate across years. They used to be
        # overwritten, so a run with 12 fallback windows spread over 4 years
        # reported only the last year's count.
        for k in ("solver", "formulation", "complementarity_reason", "last_error"):
            if diag.get(k) is not None:
                agg[k] = diag[k]
        for k in ("fallback_windows", "degenerate_windows", "milp_rescues"):
            agg[k] = int(agg.get(k, 0)) + int(diag.get(k, 0) or 0)
        agg["windows"] += int(diag.get("windows", 0))
        agg["solve_seconds"] += float(diag.get("solve_seconds", 0.0))

        all_years.append(hourly_y)
        _update_bar(bar, yi + 1, len(years), f"Optimising year {y}…")

    _close_bar(bar)
    return pd.concat(all_years, ignore_index=True), agg


# -----------------------------
# KPI comparison helper
# -----------------------------
def _optimizer_settings_rows(
    o: OptimizerParams, p: Optional[BESSParams] = None
) -> List[Dict[str, Any]]:
    """Flat, Excel-friendly dump of the optimiser settings for reproducibility."""
    rows = [
        {"Setting": "Look-ahead horizon (h)", "Value": o.horizon_hours},
        {"Setting": "Commit step (h)", "Value": o.step_hours},
        {"Setting": "Information set", "Value": o.forecast_mode},
        {"Setting": "Forecast method", "Value": o.forecast_method},
        {"Setting": "Solver", "Value": o.solver},
        {"Setting": "Time limit per window (s)", "Value": o.time_limit_s},
        {"Setting": "MIP relative gap", "Value": o.mip_gap},
        {"Setting": "Weight - peak shortfall", "Value": o.w_penalty},
        {"Setting": "Weight - curtailed energy", "Value": o.w_curtail},
        {"Setting": "Weight - BESS wear", "Value": o.w_degradation},
        {"Setting": "Grid import price", "Value": o.grid_import_price},
        {"Setting": "Tariff (hour: price)",
         "Value": ", ".join(f"{k}:{v}" for k, v in sorted((o.tariff or {}).items()))
         or "not used"},
        {"Setting": "Allow grid charging", "Value": o.allow_grid_charging},
        {"Setting": "Forbid charging in peak", "Value": o.no_charge_in_peak},
        {"Setting": "Self-discharge (%/h)",
         "Value": round(o.self_discharge_per_hour * 100.0, 4)},
        {"Setting": "Terminal SoC target (%)",
         "Value": "free" if o.terminal_soc_pct is None else o.terminal_soc_pct},
        {"Setting": "Max equivalent cycles / year",
         "Value": "unlimited" if o.max_equivalent_cycles_per_year is None
         else o.max_equivalent_cycles_per_year},
        {"Setting": "Force complementarity",
         "Value": "auto" if o.force_complementarity is None else o.force_complementarity},
        {"Setting": "Degeneracy epsilon (LP only)",
         "Value": o.degeneracy_epsilon},
        {"Setting": "Spike filter window", "Value": o.spike_filter_window},
        {"Setting": "Spike filter k", "Value": o.spike_filter_k},
    ]
    if p is not None:
        rows = [
            {"Setting": "Initial SoC (% of usable band)",
             "Value": p.initial_soc_pct},
            {"Setting": "Min SoC (%)", "Value": p.min_soc_pct},
            {"Setting": "Max SoC (%)", "Value": p.max_soc_pct},
            {"Setting": "Simulated years (control)", "Value": p.n_years},
            {"Setting": "Solar share of contract",
             "Value": p.solar_contract_share},
        ] + rows
    return rows


def kpi_summary(hourly: pd.DataFrame, p: BESSParams) -> Dict[str, float]:
    """Headline KPIs, used for the rule-based vs optimised A/B comparison."""
    if hourly is None or len(hourly) == 0:
        return {}
    gen = float(hourly["gen_MWh"].sum())
    supplied = float(hourly["Total_Supplied_MWh"].sum())
    penalty = float(hourly["Penalty_MWh"].sum())
    curtailed = float(hourly["Curtailed_MWh"].sum())
    bess_out = float(hourly["BESS_to_Grid_MWh"].sum())

    if p.mode == "single":
        loss_cols = ["BESS_Charge_Loss_MWh"]
    else:
        loss_cols = [c for c in ("solar_bess_loss_MWh", "wind_bess_loss_MWh") if c in hourly.columns]
    loss = float(sum(hourly[c].sum() for c in loss_cols)) if loss_cols else 0.0

    return {
        "Total generation (MWh)": gen,
        "Total supplied (MWh)": supplied,
        "BESS export (MWh)": bess_out,
        "Penalty (MWh)": penalty,
        "Curtailed (MWh)": curtailed,
        "BESS losses (MWh)": loss,
        "Delivery ratio (%)": (supplied / gen * 100.0) if gen > 0 else 0.0,
    }


def _optimizer_sidebar() -> Optional[OptimizerParams]:
    """
    Render the optimiser controls and build an `OptimizerParams`.

    Kept out of `main()` so the sidebar stays readable. Returns None when the
    required solver backend is missing (the caller then shows an error).
    """
    st.sidebar.header("Optimiser settings")

    if not PULP_AVAILABLE:
        st.sidebar.error(
            "PuLP is not installed, so the MILP optimiser is unavailable.\n\n"
            "Install it with `pip install pulp` (CBC ships with PuLP) or "
            "`pip install highspy`, or switch back to the rule-based engine."
        )
        return None

    if not available_optimization_solvers():
        st.sidebar.error(
            "PuLP is installed but **no solver backend is available**, so the "
            "MILP optimiser cannot run.\n\n"
            "`pip install highspy` (HiGHS) is the quickest fix, or reinstall "
            "PuLP so that it ships the CBC binary."
        )
        return None

    installed = available_optimization_solvers()
    st.sidebar.caption(f"Solvers available: {', '.join(installed) or 'none'}")

    # ---------------------------------------------------------------------
    # ALIGNMENT RULE -- do not "tidy" these labels back into long ones.
    #
    # `st.columns` lays its columns out TOP-aligned, and a Streamlit input is a
    # vertical stack of [label] then [control]. If one label in a pair wraps to
    # two lines and its partner does not, the two controls start at different
    # y positions and the row visibly steps up and down.
    #
    # `vertical_alignment="center"` does NOT fix it: that centres whole blocks,
    # so the taller two-line block ends up with its control LOWER than the
    # shorter one. Same defect, mirrored.
    #
    # The only robust fix is to keep every label inside a column pair on ONE
    # line. Detail therefore lives in the help tooltip and in a caption above
    # the pair, not in the label itself.
    # `_shots.py` asserts this geometrically so a regression fails loudly.
    # ---------------------------------------------------------------------
    st.sidebar.subheader("Information set")
    forecast_label = st.sidebar.radio(
        "Generation the optimiser may use",
        options=[
            "Causal forecast (operational)",
            "Realised generation (perfect-foresight benchmark)",
        ],
        index=0,
        help=(
            "Causal: every window is solved on a forecast built only from data "
            "strictly before the window starts (seasonal-naive, 24 h lag). The "
            "committed hours are then settled against the realised generation. "
            "Realised: the optimiser sees the actual series for the whole "
            "window - a perfect-foresight upper bound, not an operational "
            "controller, and not achievable in practice."
        ),
    )
    forecast_mode = "causal" if forecast_label.startswith("Causal") else "oracle"

    if forecast_mode == "causal":
        _method_label = st.sidebar.selectbox(
            "Forecast method",
            options=["Seasonal-naive (24 h lag)", "Persistence (last value)"],
            index=0,
            help=(
                "Seasonal-naive repeats the same hour of the previous day. "
                "Persistence repeats the last observed value for the whole "
                "horizon. Both are causal; neither reads future data."
            ),
        )
        forecast_method = (
            "seasonal_naive" if _method_label.startswith("Seasonal") else "persistence"
        )
    else:
        forecast_method = "seasonal_naive"
        st.sidebar.warning(
            "Perfect-foresight mode: results are a benchmark upper bound, not an "
            "operational dispatch."
        )

    st.sidebar.subheader("Receding horizon")
    commit_label = st.sidebar.radio(
        "Commit mode",
        options=[
            "Receding horizon (commit 1 h)",
            "Block commitment (commit N h)",
        ],
        index=0,
        help=(
            "Receding horizon re-solves every hour and locks in only that hour, "
            "so the next decision uses the state actually reached. Block "
            "commitment locks in N hours before re-solving; it is faster but "
            "less responsive to forecast error."
        ),
    )
    receding = commit_label.startswith("Receding")

    st.sidebar.caption("Rolling horizon: solve N hours, commit M hours")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        horizon_hours = st.number_input(
            "Horizon (h)", min_value=6, max_value=168,
            value=24, step=6,
            help="Look-ahead window solved at every step, on the causal forecast. "
                 "24 h covers the diurnal cycle and the default peak blocks.",
        )
    with c2:
        if receding:
            st.number_input(
                "Commit (h)", min_value=1, max_value=1,
                value=1, step=1, disabled=True,
                help="Receding horizon always commits exactly one hour.",
            )
            step_hours = 1
        else:
            step_hours = st.number_input(
                "Commit (h)", min_value=1, max_value=168,
                value=6, step=1,
                help="Only this many hours of the solution are locked in before "
                     "the model is re-solved with the settled SoC.",
            )
    if step_hours > horizon_hours:
        st.sidebar.warning("Commit step is larger than the horizon; it will be clamped.")
        step_hours = horizon_hours
    st.sidebar.caption(
        f"About {max(len(range(0, 8760, int(step_hours))), 1):,} solves per "
        f"simulated year at a {int(step_hours)}-hour commit."
    )

    solver_label = st.sidebar.selectbox(
        "Solver", options=["auto"] + installed,
        index=0,
        help="auto prefers CBC, then HiGHS, then SCIP/GLPK.",
    )

    c3, c4 = st.sidebar.columns(2)
    with c3:
        time_limit_s = st.number_input(
            "Time limit (s)", min_value=1.0, max_value=600.0,
            value=30.0, step=5.0,
            help="Wall-clock budget for each individual optimisation window.",
        )
    with c4:
        mip_gap = st.number_input(
            "MIP gap", min_value=0.0, max_value=0.5,
            value=0.0, step=0.01, format="%.3f",
            help="Relative optimality gap for the MIP. 0 = exact optimum; "
                 "0.01 is normally plenty for dispatch.",
        )

    st.sidebar.subheader("Objective weights")
    st.sidebar.caption(
        "Currency per MWh. A large shortfall weight keeps the contract "
        "guarantee ahead of everything else."
    )
    c5, c6 = st.sidebar.columns(2)
    with c5:
        w_penalty = st.number_input(
            "Shortfall", min_value=0.0, max_value=1e6,
            value=1000.0, step=50.0,
            help="Cost per MWh of peak-hour contract shortfall (Penalty_MWh).",
        )
    with c6:
        w_curtail = st.number_input(
            "Curtailment", min_value=0.0, max_value=1e6,
            value=1.0, step=0.5,
            help="Cost per MWh of curtailed generation (Curtailed_MWh).",
        )
    c7, c8 = st.sidebar.columns(2)
    with c7:
        w_degradation = st.number_input(
            "Wear cost", min_value=0.0, max_value=1e4,
            value=0.0, step=0.5,
            help="Cost per MWh of BESS throughput (charge + discharge/eta_dis).",
        )
    with c8:
        grid_import_price = st.number_input(
            "Import price", min_value=-1000.0, max_value=10000.0,
            value=0.0, step=5.0,
            help="Cost per MWh paid for grid energy used to charge the BESS. "
                 "Only used when grid charging is enabled.",
        )

    with st.sidebar.expander("Optional arbitrage tariff", expanded=False):
        st.caption(
            "Enables energy arbitrage on top of the contract objective. Enter "
            "24 prices (0-23). Leave the first two rows empty to disable it."
        )
        tariff_enabled = st.checkbox("Use a time-of-use tariff", value=False)
        tariff_input = ""
        if tariff_enabled:
            tariff_input = st.text_area(
                "Price per hour (comma separated, 0-23)",
                value=",".join(
                    str(x) for x in
                    ([0, 0] + [20] * 7 + [60] * 7 + [20] * 4 + [140] * 4)
                ),
                height=90,
            )
        tariff = None
        if tariff_enabled and tariff_input.strip():
            parts = [p.strip() for p in tariff_input.replace("\n", ",").split(",")]
            if len(parts) != 24:
                st.error("Tariff must contain exactly 24 values (hours 0-23).")
                tariff = None
            else:
                try:
                    vals = [float(p) for p in parts]
                    tariff = {h: vals[h] for h in range(24)}
                    st.success("Tariff loaded.")
                except ValueError:
                    st.error("Tariff contains non-numeric values.")
                    tariff = None

    st.sidebar.subheader("Operating options")
    allow_grid_charging = st.sidebar.checkbox(
        "Allow charging from the grid",
        value=False,
        help="Off by default, which keeps the optimiser comparable with the "
             "rule-based engine (which can only charge from surplus generation).",
    )
    no_charge_in_peak = st.sidebar.checkbox(
        "Forbid charging during peak hours (same as rule-based engine)",
        value=True,
    )
    use_terminal = st.sidebar.checkbox(
        "Fix the state of charge at the end of each window", value=False
    )
    terminal_soc_pct = None
    if use_terminal:
        terminal_soc_pct = st.sidebar.slider(
            "End-of-window SOC target (%)", 0.0, 100.0, 90.0, 1.0
        )

    use_cycles = st.sidebar.checkbox(
        "Limit equivalent full cycles per year", value=False
    )
    max_cycles = None
    if use_cycles:
        max_cycles = st.sidebar.number_input(
            "Equivalent full cycles / year", min_value=1.0, max_value=2000.0,
            value=400.0, step=25.0,
            help="Throughput budget: sum(charge + discharge/eta_dis) <= "
                 "2 * capacity * cycles. Prorated to each window.",
        )

    self_discharge = st.sidebar.number_input(
        "Self-discharge (%/hour)", min_value=0.0, max_value=5.0,
        value=0.0, step=0.05, format="%.2f",
        help="Storage self-discharge rho, applied every hour in the SoC balance.",
    )

    with st.sidebar.expander("Numerical formulation", expanded=False):
        force_choice = st.radio(
            "Complementarity",
            ["Auto (recommended)", "Force MILP", "Force LP"],
            index=0,
            help=(
                "Auto adds the charge/discharge binaries only when they are "
                "provably required (RTE = 1 or a negative price signal), which "
                "makes the problem a pure LP. See Prat, Lusby & Pinson, "
                "arXiv:2508.12999, Proposition 1."
            ),
        )
        force_complementarity = {"Auto (recommended)": None, "Force MILP": True, "Force LP": False}[force_choice]
        degeneracy_epsilon = st.number_input(
            "Degeneracy epsilon (LP only)",
            min_value=0.0, max_value=1.0, value=1e-6, step=1e-6, format="%.6f",
            help="Vanishing throughput cost that stops the simplex returning a "
                 "vertex with simultaneous charge and discharge. Set to 0 to "
                 "disable and rely on the automatic MILP re-solve instead.",
        )
        solver_verbose = st.checkbox("Verbose solver log", value=False)

    with st.sidebar.expander("Price spike filter (tariff only)", expanded=False):
        st.caption(
            "Causal (trailing) robust clip of the tariff. It is never applied "
            "to the contract penalty, and it never looks ahead."
        )
        spike_window = st.number_input(
            "Trailing window (0 = off)", min_value=0, max_value=48, value=0, step=1
        )
        spike_k = st.number_input("k (std deviations)", min_value=0.5, max_value=10.0,
                                  value=2.5, step=0.5)

    return OptimizerParams(
        horizon_hours=int(horizon_hours),
        step_hours=int(step_hours),
        forecast_mode=forecast_mode,
        forecast_method=forecast_method,
        solver=solver_label,
        time_limit_s=float(time_limit_s),
        mip_gap=float(mip_gap),
        msg=bool(solver_verbose),
        w_penalty=float(w_penalty),
        w_curtail=float(w_curtail),
        w_degradation=float(w_degradation),
        grid_import_price=float(grid_import_price),
        tariff=tariff,
        allow_grid_charging=bool(allow_grid_charging),
        no_charge_in_peak=bool(no_charge_in_peak),
        self_discharge_per_hour=float(self_discharge) / 100.0,
        terminal_soc_pct=terminal_soc_pct,
        max_equivalent_cycles_per_year=max_cycles,
        force_complementarity=force_complementarity,
        degeneracy_epsilon=float(degeneracy_epsilon),
        spike_filter_window=int(spike_window),
        spike_filter_k=float(spike_k),
    )


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Hybrid BESS Dispatch Tool", layout="wide")

    import base64

    BG_PATH = Path(
        r"D:\\VAYUMITHRA\\BESS\\BESS Tool\\vayumithra_wallpaper.png"
    )
    if BG_PATH.exists():
        try:
            encoded = base64.b64encode(BG_PATH.read_bytes()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{encoded}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    st.title("Hybrid PV + Wind + BESS dispatch tool")

    with st.sidebar:
        logo_path = Path(VAYUMITHRA_LOGO_PATH)
        if logo_path.exists():
            try:
                st.image(str(logo_path), width=140)
            except Exception:
                pass
        st.markdown(f"[Website]({VAYUMITHRA_WEBSITE})", unsafe_allow_html=False)
        st.markdown("**Vayumithra – Renewable Analytics Suite**")

    # ---------------- Dispatch engine (FIRST control in the sidebar) ----------------
    # This sits at the very top on purpose: it decides which algorithm runs the
    # dispatch, and the optimiser's settings panel is rendered immediately below
    # it so the choice and its consequences are always visible together.
    st.sidebar.header("⚡ BESS dispatch engine")
    dispatch_engine = st.sidebar.radio(
        "Dispatch engine",
        ["Rule-based (deterministic)", "MILP rolling-horizon optimiser"],
        index=0,
        help=(
            "**Rule-based (deterministic)** — the original greedy engine. "
            "Charges from surplus generation, exports the rest, and discharges "
            "into the peak-hour contract shortfall. Fully deterministic and "
            "instant.\n\n"
            "**MILP rolling-horizon optimiser** — solves a mixed-integer / LP "
            "dispatch problem over a rolling look-ahead horizon (default 48 h), "
            "commits only the first step, then re-solves with the realised state "
            "of charge. Supports grid charging, time-of-use tariffs, cycle "
            "budgets, terminal SoC targets and self-discharge."
        ),
    )
    use_optimizer = dispatch_engine.startswith("MILP")

    optimizer_params = None
    if use_optimizer:
        optimizer_params = _optimizer_sidebar()
    else:
        st.sidebar.caption(
            "Optimiser settings appear here when the MILP engine is selected."
        )

    st.sidebar.divider()

    # ---------------- Sidebar inputs ----------------
    st.sidebar.header("Project")
    project_name = st.sidebar.text_input(
        "Project name",
        value="MyProject",
        help="This will be used in the Excel export file name",
    )

    
    st.sidebar.header("Plant & contract")
    solar_cap_mw = st.sidebar.number_input(
        "Solar plant capacity (MWac)", value=281.0, step=1.0
    )
    wind_cap_mw = st.sidebar.number_input(
        "Wind plant capacity (MWac)", value=0.0, step=1.0
    )
    contract_mw = st.sidebar.number_input(
        "Contract capacity (MW for penalty and peak demand)",
        value=190.0,
        step=1.0,
    )

    st.sidebar.header("Configuration mode")
    config_mode = st.sidebar.selectbox(
        "Mode",
        ["single", "split", "custom"],
    )

    if config_mode == "custom":
        base_engine = st.sidebar.radio(
            "Custom base logic", ["Single BESS base", "Split BESS base"], index=0
        )
        engine_mode = "single" if base_engine.startswith("Single") else "split"
    else:
        engine_mode = config_mode

    # FLAW 6 FIX - the split-mode contract split was hardcoded at 84/16 with no
    # way to change it. It is now a sidebar control, shown only in split mode.
    if engine_mode == "split":
        solar_contract_share = st.sidebar.slider(
            "Solar share of contract capacity (%)",
            min_value=0,
            max_value=100,
            value=84,
            step=1,
            help=(
                "Split mode only: share of the contract MW attributed to the "
                "solar plant. The wind plant carries the remainder. Each plant "
                "is penalised against its own share in peak hours, so a surplus "
                "in one cannot hide a shortfall in the other."
            ),
        ) / 100.0
    else:
        solar_contract_share = 1.0  # unused when a single BESS is configured

    st.sidebar.header("BESS parameters")

    if engine_mode == "single":
        bess_energy_mwh = st.sidebar.number_input(
            "Single BESS energy (MWh)", value=600.0, step=10.0
        )
        pcs_power_mw = st.sidebar.number_input(
            "Single BESS PCS power (MW)", value=200.0, step=10.0
        )
        solar_bess_energy_mwh = solar_pcs_power_mw = 0.0
        wind_bess_energy_mwh = wind_pcs_power_mw = 0.0
    else:
        bess_energy_mwh = pcs_power_mw = 0.0
        solar_bess_energy_mwh = st.sidebar.number_input(
            "Solar BESS energy (MWh)", value=160.0, step=10.0
        )
        solar_pcs_power_mw = st.sidebar.number_input(
            "Solar BESS PCS (MW)", value=160.0, step=10.0
        )
        wind_bess_energy_mwh = st.sidebar.number_input(
            "Wind BESS energy (MWh)", value=30.0, step=5.0
        )
        wind_pcs_power_mw = st.sidebar.number_input(
            "Wind BESS PCS (MW)", value=30.0, step=5.0
        )

    st.sidebar.header("Battery & operating band")

    # FLAW 4 FIX - the initial state of charge was implicitly "full", which
    # injected free stored energy at t=0. It is now explicit and user-controlled.
    initial_soc_pct = st.sidebar.slider(
        "Initial state of charge (% of usable band)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help=(
            "Energy in the battery before the first simulated hour, as a "
            "percentage of the usable band (Min SOC to Max SOC). 0 starts at "
            "the floor, 100 at the ceiling. The old implicit assumption was "
            "100 %, which created free energy in the first hours; 50 % is the "
            "neutral default. The end-of-horizon stock is reported so any "
            "boundary effect can be checked."
        ),
    )
    min_soc_pct = st.sidebar.slider("Min SOC (%)", 0.0, 50.0, 0.0, 1.0)
    max_soc_pct = st.sidebar.slider("Max SOC (%)", 50.0, 100.0, 90.0, 1.0)
    peak_hours_str = st.sidebar.text_input(
        "Peak hours (0–23, comma-separated)", "5,6,19,20"
    )

    st.sidebar.header("Grid export limit")
    grid_limit_mw = st.sidebar.number_input(
        "Grid supply limit (MW). Use a large number for 'unlimited'; 0 means fully blocked.",
        value=0.0,
        step=10.0,
    )
    no_grid_limit = st.sidebar.checkbox(
        "No grid export limit (unlimited)", value=True
    )
    if no_grid_limit:
        grid_limit_param = None
    else:
        grid_limit_param = grid_limit_mw

    st.sidebar.header("Battery reset behaviour")
    reset_battery_each_year = st.sidebar.checkbox(
        "Reset battery to the initial state of charge at the start of every year "
        "(unrealistic; off by default)",
        value=False,
    )

    n_years = st.sidebar.number_input(
        "Number of years (repeat same TS)",
        value=1,
        min_value=1,
        max_value=25,
        step=1,
        help=(
            "Length of the simulated horizon. The uploaded time series is "
            "repeated (or truncated) to exactly this many calendar years, "
            "starting from the first year in the file. Per-year degradation and "
            "year-dependent efficiencies follow this project-year order."
        ),
    )

    st.sidebar.header("Degradation profiles")
    degr_single_file = degr_solar_file = degr_wind_file = None
    if engine_mode == "single":
        degr_single_file = st.sidebar.file_uploader(
            "Single BESS degradation profile", key="single_degr"
        )
    else:
        degr_solar_file = st.sidebar.file_uploader(
            "Solar BESS degradation profile", key="solar_degr"
        )
        degr_wind_file = st.sidebar.file_uploader(
            "Wind BESS degradation profile", key="wind_degr"
        )

    st.sidebar.header("RTE configuration")
    rte_mode = st.sidebar.radio(
        "Efficiency mode",
        ["Constant efficiencies", "Year-dependent efficiencies"],
        index=0,
    )

    rte_by_year = None
    if rte_mode == "Constant efficiencies":
        eta_charge = st.sidebar.slider(
            "Charge efficiency", 0.90, 1.00, 0.97, 0.005
        )
        eta_discharge = st.sidebar.slider(
            "Discharge efficiency", 0.90, 1.00, 0.96, 0.005
        )
    else:
        years = list(range(1, n_years + 1))
        st.sidebar.markdown(
            "Edit per‑year efficiencies below (in %). Rows follow simulated years (1..N)."
        )
        default_df = pd.DataFrame(
            {
                "Charge efficiency (%)": [93.0] * len(years),
                "Discharge efficiency (%)": [93.0] * len(years),
            },
            index=years,
        )
        default_df.index.name = "Year"
        rte_df = st.sidebar.data_editor(
            default_df,
            use_container_width=True,
            num_rows="fixed",
            key="rte_by_year_editor",
        )

        eta_ch_list = (rte_df["Charge efficiency (%)"] / 100.0).tolist()
        eta_dis_list = (rte_df["Discharge efficiency (%)"] / 100.0).tolist()

        rte_by_year = pd.DataFrame(
            {"eta_charge": eta_ch_list, "eta_discharge": eta_dis_list},
            index=years,
        )

        eta_charge = float(eta_ch_list[0])
        eta_discharge = float(eta_dis_list[0])

    uploaded = st.file_uploader(
        "Upload BESSTOOL_INPUT (Year,Day,Month,Hour,solar_MWh,wind_MWh)",
        type=["xlsx", "csv"],
    )
    if uploaded is None:
        engine_label = (
            "MILP rolling-horizon optimiser"
            if use_optimizer
            else "Rule-based (deterministic) engine"
        )
        st.info(
            f"**Dispatch engine: {engine_label}**\n\n"
            "Upload a `BESSTOOL_INPUT` file with the columns "
            "`Year, Day, Month, Hour, solar_MWh, wind_MWh` to run the simulation."
        )
        st.markdown(
            "\nUpload the BESSTOOL_INPUT file to run the simulation.... ",
            unsafe_allow_html=True,
        )
        if not use_optimizer:
            st.caption(
                "Tip: switch **Dispatch engine** in the sidebar (top) to "
                "*MILP rolling-horizon optimiser* to dispatch with the solver "
                "instead of the greedy rule."
            )
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)
    except Exception as exc:
        st.error("**Could not read the uploaded file.**")
        st.caption(f"{type(exc).__name__}: {exc}")
        st.info(
            "Upload a non-empty CSV or XLSX file with the columns "
            "`Year, Day, Month, Hour, solar_MWh, wind_MWh`. "
            "Nothing was simulated."
        )
        return

    # FIX 4 - validate before anything downstream touches the frame.
    df_in, input_errors, input_warnings = validate_input_frame(df_in)

    if input_errors:
        st.error("**The input file does not satisfy the BESSTOOL_INPUT contract.**")
        for _err in input_errors:
            st.markdown(f"- {_err}")
        st.caption(
            "Required columns: `Year, Day, Month, Hour, solar_MWh, wind_MWh` - "
            "one row per hour, `Hour` in 0..23. Nothing was simulated."
        )
        return

    for _warn in input_warnings:
        st.warning(_warn)

    # Optional P50/P75/P90 generation scenarios.  A file without these
    # columns is unaffected: the selector is not shown and the realised
    # solar_MWh/wind_MWh columns are used exactly as before.
    scenario_map = available_generation_scenarios(df_in)
    scenario_source = "file"

    # Single-file build: if the upload is a plain P50 file, derive P75/P90
    # here instead of requiring a pre-built scenario file.
    if not scenario_map:
        _has_basis = (
            "solar_MWh" in df_in.columns and "wind_MWh" in df_in.columns
        ) or ("solar_p50_MWh" in df_in.columns and "wind_p50_MWh" in df_in.columns)
        if _has_basis:
            _mode_label = st.sidebar.radio(
                "Scenario conversion",
                options=[
                    "Annual factor (bankability)",
                    "Hourly residual (stress test)",
                ],
                index=0,
                help="Annual: scale each technology's P50 by its own fitted "
                     "annual energy factor and keep the hourly shape. Hourly: add "
                     "the month-hour residual quantile in every hour - a "
                     "deliberately severe joint stress test.",
            )
            _mode = "annual" if _mode_label.startswith("Annual") else "hourly"
            try:
                df_in = build_generation_scenarios(df_in, mode=_mode)
                scenario_map = available_generation_scenarios(df_in)
                scenario_source = "embedded " + _mode
                st.sidebar.caption(
                    "P75/P90 built from P50 with the embedded "
                    + _mode
                    + " calibration ("
                    + (
                        SCENARIO_ANNUAL_YEARS
                        if _mode == "annual"
                        else SCENARIO_HOURLY_YEARS
                    )
                    + ")."
                )
            except Exception as _exc:
                st.sidebar.warning("Could not build P75/P90 from P50: " + str(_exc))

    if scenario_map:
        options = sorted(scenario_map)
        default_level = 50 if 50 in options else options[0]
        chosen_level = st.sidebar.radio(
            "Generation scenario",
            options=options,
            index=options.index(default_level),
            format_func=lambda level: f"P{level}",
            help="P90 is the 90 % probability-of-exceedance case (the "
                 "conservative low-generation case), not the 90th percentile. "
                 + (
                     "Built from the uploaded P50 by the embedded calibration."
                     if scenario_source != "file"
                     else "The file was produced by quantile_scenarios.py."
                 ),
        )
        df_in, applied = apply_generation_scenario(df_in, chosen_level)
        st.sidebar.caption(
            f"Using P{chosen_level} for "
            + ", ".join(sorted(applied))
            + ". Hours outside the scenario window keep realised generation; "
              "the original columns remain in the file for audit."
        )

    st.subheader("Input data preview")
    preview = df_in.head(24)
    numeric_cols = preview.select_dtypes(include="float").columns.tolist()
    st.dataframe(
        preview.style.format("{:.2f}", na_rep="-", subset=numeric_cols)
    )

    input_years = sorted(int(v) for v in df_in["Year"].unique())
    n_years_from_data = len(input_years)

    # FLAW 3 FIX - build the frame that will actually be simulated. The
    # sidebar control now determines the horizon; the uploaded series is
    # repeated or truncated to match it.
    df_run, years_note = expand_to_n_years(df_in, int(n_years))
    st.caption(f"**Simulated horizon:** {years_note}")
    if n_years < n_years_from_data:
        st.warning(
            "The uploaded file contains more calendar years than the "
            "**Number of years (repeat same TS)** control. Only the first "
            f"{int(n_years)} year(s) will be dispatched."
        )

    formulas_df = None
    profile_name = None
    profiles = load_all_profiles()

    if config_mode == "custom":
        st.sidebar.header("Custom formula profile")

        if engine_mode not in profiles or not isinstance(profiles[engine_mode], dict):
            profiles[engine_mode] = {}

        if engine_mode == "single":
            st.sidebar.markdown("Base: **Single BESS** (shared BESS).")
            profile_suffix = "_single"
            default_formulas = pd.DataFrame(
                {
                    "New column": [
                        "gen_MWh",
                        "is_peak",
                        "BESS_Capacity_MWh",
                        "BESS_SOC_MWh",
                        "BESS_SOC_pct",
                        "BESS_Charge_in_MWh",
                        "BESS_Charge_to_SOC_MWh",
                        "BESS_Charge_Loss_MWh",
                        "BESS_Discharge_MWh",
                        "BESS_TotalCycles",
                        "solar_to_grid_MWh",
                        "wind_to_grid_MWh",
                        "GenPlant_to_Grid_MWh",
                        "BESS_to_Grid_MWh",
                        "Total_Supplied_MWh",
                        "Peak_Time_Energy_Supplied_MWh",
                        "Offpeak_Supplied_MWh",
                        "Curtailed_MWh",
                        "Penalty_MWh",
                        "Custom_KPI",
                    ],
                    "Formula": [
                        "gen_MWh",
                        "is_peak",
                        "BESS_Capacity_MWh",
                        "BESS_SOC_MWh",
                        "BESS_SOC_pct",
                        "BESS_Charge_in_MWh",
                        "BESS_Charge_to_SOC_MWh",
                        "BESS_Charge_in_MWh - BESS_Charge_to_SOC_MWh",
                        "BESS_Discharge_MWh",
                        "BESS_TotalCycles",
                        "solar_to_grid_MWh",
                        "wind_to_grid_MWh",
                        "solar_to_grid_MWh + wind_to_grid_MWh",
                        "BESS_to_Grid_MWh",
                        "Total_Supplied_MWh",
                        "np.where(is_peak == 1, Total_Supplied_MWh, 0)",
                        "np.where(is_peak == 0, Total_Supplied_MWh, 0)",
                        "Curtailed_MWh",
                        "np.where(is_peak == 1, np.maximum(contract_mw - Total_Supplied_MWh, 0), 0)",
                        "Total_Supplied_MWh - Penalty_MWh",
                    ],
                }
            )
            editor_key = "custom_formulas_single"
        else:
            st.sidebar.markdown("Base: **Split BESS** (solar + wind, separate BESS).")
            profile_suffix = "_split"
            default_formulas = pd.DataFrame(
                {
                    "New column": [
                        "gen_MWh",
                        "is_peak",
                        "solar_bess_capacity_MWh",
                        "solar_bess_soc_MWh",
                        "solar_bess_soc_pct",
                        "solar_bess_charge_in_MWh",
                        "solar_bess_charge_to_soc_MWh",
                        "solar_bess_loss_MWh",
                        "solar_bess_discharge_MWh",
                        "solar_bess_total_cycles",
                        "wind_bess_capacity_MWh",
                        "wind_bess_soc_MWh",
                        "wind_bess_soc_pct",
                        "wind_bess_charge_in_MWh",
                        "wind_bess_charge_to_soc_MWh",
                        "wind_bess_loss_MWh",
                        "wind_bess_discharge_MWh",
                        "wind_bess_total_cycles",
                        "solar_to_grid_MWh",
                        "solar_to_bess_MWh",
                        "wind_to_grid_MWh",
                        "wind_to_bess_MWh",
                        "GenPlant_to_Grid_MWh",
                        "BESS_to_Grid_MWh",
                        "Total_Supplied_MWh",
                        "Peak_Time_Energy_Supplied_MWh",
                        "Offpeak_Supplied_MWh",
                        "Curtailed_MWh",
                        "Penalty_MWh",
                        "Solar_Wind_Ratio",
                    ],
                    "Formula": [
                        "gen_MWh",
                        "is_peak",
                        "solar_bess_capacity_MWh",
                        "solar_bess_soc_MWh",
                        "solar_bess_soc_pct",
                        "solar_bess_charge_in_MWh",
                        "solar_bess_charge_to_soc_MWh",
                        "solar_bess_charge_in_MWh - solar_bess_charge_to_soc_MWh",
                        "solar_bess_discharge_MWh",
                        "solar_bess_total_cycles",
                        "wind_bess_capacity_MWh",
                        "wind_bess_soc_MWh",
                        "wind_bess_soc_pct",
                        "wind_bess_charge_in_MWh",
                        "wind_bess_charge_to_soc_MWh",
                        "wind_bess_charge_in_MWh - wind_bess_charge_to_soc_MWh",
                        "wind_bess_discharge_MWh",
                        "wind_bess_total_cycles",
                        "solar_to_grid_MWh",
                        "solar_to_bess_MWh",
                        "wind_to_grid_MWh",
                        "wind_to_bess_MWh",
                        "GenPlant_to_Grid_MWh",
                        "BESS_to_Grid_MWh",
                        "Total_Supplied_MWh",
                        "np.where(is_peak == 1, Total_Supplied_MWh, 0)",
                        "np.where(is_peak == 0, Total_Supplied_MWh, 0)",
                        "Curtailed_MWh",
                        "np.where(is_peak == 1, np.maximum(contract_mw - Total_Supplied_MWh, 0), 0)",
                        "solar_to_grid_MWh / (wind_to_grid_MWh + 1e-6)",
                    ],
                }
            )
            editor_key = "custom_formulas_split"

        profile_name = st.sidebar.text_input(
            "Profile name",
            value=f"MyProfile{profile_suffix}",
        )
        load_profile_btn = st.sidebar.button("Load profile")
        save_profile_btn = st.sidebar.button("Save profile")

        current_formulas = default_formulas.copy()
        if load_profile_btn:
            prof_group = profiles.get(engine_mode, {})
            if profile_name in prof_group:
                try:
                    current_formulas = pd.DataFrame(prof_group[profile_name])
                    st.sidebar.success(
                        f"Profile '{profile_name}' loaded for {engine_mode}."
                    )
                except Exception as e:
                    st.sidebar.warning(
                        f"Could not load profile '{profile_name}': {e}"
                    )
            else:
                st.sidebar.warning(
                    f"No profile named '{profile_name}' for {engine_mode}."
                )

        formulas_df = st.sidebar.data_editor(
            current_formulas,
            num_rows="dynamic",
            use_container_width=True,
            key=editor_key,
        )

        if save_profile_btn:
            profiles[engine_mode][profile_name] = formulas_df.to_dict(orient="list")
            save_all_profiles(profiles)
            st.sidebar.success(f"Profile '{profile_name}' saved for {engine_mode}.")

    if st.button("Run BESS dispatch"):
        try:
            params = BESSParams(
                solar_cap_mw=solar_cap_mw,
                wind_cap_mw=wind_cap_mw,
                contract_mw=contract_mw,
                bess_energy_mwh=bess_energy_mwh,
                pcs_power_mw=pcs_power_mw,
                solar_bess_energy_mwh=solar_bess_energy_mwh,
                solar_pcs_power_mw=solar_pcs_power_mw,
                wind_bess_energy_mwh=wind_bess_energy_mwh,
                wind_pcs_power_mw=wind_pcs_power_mw,
                eta_charge=eta_charge,
                eta_discharge=eta_discharge,
                min_soc_pct=min_soc_pct,
                max_soc_pct=max_soc_pct,
                initial_soc_pct=initial_soc_pct,
                peak_hours=parse_peak_hours(peak_hours_str),
                grid_limit_mw=grid_limit_param,
                mode=engine_mode,
                solar_contract_share=solar_contract_share,
                n_years=int(n_years),
                rte_by_year=rte_by_year,
                reset_battery_each_year=reset_battery_each_year,
            )

            degr_single = (
                load_degradation_profile(degr_single_file)
                if engine_mode == "single"
                else pd.Series(1.0, index=range(1, 26))
            )
            degr_solar = (
                load_degradation_profile(degr_solar_file)
                if engine_mode == "split"
                else pd.Series(1.0, index=range(1, 26))
            )
            degr_wind = (
                load_degradation_profile(degr_wind_file)
                if engine_mode == "split"
                else pd.Series(1.0, index=range(1, 26))
            )

            optimizer_diag = None
            baseline_hourly = None

            if use_optimizer:
                if optimizer_params is None:
                    st.error(
                        "The MILP optimiser needs a solver backend. Install "
                        "`pulp` (bundles CBC) or `highspy`, or switch the dispatch "
                        "engine back to 'Rule-based (deterministic)'."
                    )
                    return

                # Surface any configuration problem before a long run.
                if not params.peak_hours:
                    st.warning(
                        "No peak hours are defined, so the contract penalty is "
                        "identically zero and the optimiser has no reliability "
                        "signal to work with. Add peak hours or enable a tariff."
                    )

                st.info(
                    f"Running the MILP rolling-horizon optimiser "
                    f"(horizon {optimizer_params.horizon_hours} h, "
                    f"commit {optimizer_params.step_hours} h, "
                    f"solver {optimizer_params.solver}). "
                    "This solves one optimisation per committed step and can "
                    "take considerably longer than the rule-based engine."
                )

                hourly, optimizer_diag = simulate_optimized_multi_year(
                    df_run, params, optimizer_params, degr_single, degr_solar, degr_wind
                )

                if optimizer_diag.get("fallback_windows", 0) > 0:
                    st.warning(
                        f"{optimizer_diag['fallback_windows']} of "
                        f"{optimizer_diag['windows']} optimisation windows could "
                        "not be solved and fell back to the rule-based logic."
                        + (
                            f" Last error: {optimizer_diag['last_error']}"
                            if optimizer_diag.get("last_error")
                            else ""
                        )
                    )
                if optimizer_diag.get("milp_rescues", 0) > 0:
                    st.caption(
                        f"{optimizer_diag['milp_rescues']} degenerate LP window(s) "
                        "were automatically re-solved with the complementarity "
                        "binaries so no simultaneous charge/discharge is scheduled."
                    )
                if use_optimizer and st.checkbox(
                    "Also run the rule-based engine for comparison", value=True
                ):
                    baseline_hourly = simulate_multi_year(
                        df_run, params, degr_single, degr_solar, degr_wind
                    )
            else:
                hourly = simulate_multi_year(
                    df_run, params, degr_single, degr_solar, degr_wind
                )

            if config_mode == "custom" and formulas_df is not None:
                hourly = apply_custom_formulas(hourly, formulas_df, params)

            daily = make_daily_penalty(hourly, params)
            monthly_gen, monthly_pen = make_monthly_tables(hourly, params)
            yearly = make_yearly_table(hourly, params)

            if optimizer_diag is not None:
                st.subheader("Optimiser diagnostics")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Solver", str(optimizer_diag.get("solver", "-")))
                d2.metric("Formulation", str(optimizer_diag.get("formulation", "-")))
                d3.metric(
                    "Windows solved",
                    f"{optimizer_diag.get('windows', 0) - optimizer_diag.get('fallback_windows', 0)}"
                    f" / {optimizer_diag.get('windows', 0)}",
                )
                d4.metric("Solve time", f"{optimizer_diag.get('solve_seconds', 0.0):.1f} s")
                st.caption(
                    f"Complementarity decision: {optimizer_diag.get('complementarity_reason', '-')}. "
                    f"Degenerate windows re-solved as MILP: {optimizer_diag.get('milp_rescues', 0)}."
                )

            if baseline_hourly is not None:
                st.subheader("Rule-based vs optimised comparison")
                k_opt = kpi_summary(hourly, params)
                k_base = kpi_summary(baseline_hourly, params)
                rows = []
                for k in k_opt:
                    rows.append(
                        {
                            "KPI": k,
                            "Rule-based": k_base.get(k, float("nan")),
                            "Optimised": k_opt[k],
                            "Delta": k_opt[k] - k_base.get(k, float("nan")),
                        }
                    )
                cmp_df = pd.DataFrame(rows)
                st.dataframe(
                    cmp_df.style.format(
                        "{:.2f}", na_rep="-", subset=["Rule-based", "Optimised", "Delta"]
                    )
                )

                pen_base = k_base.get("Penalty (MWh)", 0.0)
                pen_opt = k_opt.get("Penalty (MWh)", 0.0)
                if pen_opt < pen_base - 1e-6:
                    st.success(
                        f"Optimised engine removes {(pen_base - pen_opt):,.1f} MWh "
                        f"of peak shortfall ({(1 - pen_opt / pen_base) * 100:.1f}% lower)."
                    )
                elif pen_opt > pen_base + 1e-6:
                    st.warning(
                        f"Optimised engine is {(pen_opt - pen_base):,.1f} MWh WORSE "
                        "on peak shortfall. Check the optimiser settings."
                    )
                else:
                    st.info(
                        "Identical peak shortfall on both engines. With the default "
                        "options (charge only from surplus, never in peak) the "
                        "greedy rule is already near-optimal for this objective, so "
                        "the optimiser normally ties it. It starts to win once you "
                        "enable grid charging, an arbitrage tariff, a cycle budget "
                        "or a terminal SoC target."
                    )

            st.subheader("Hourly results (first 48 rows)")
            _h = hourly.head(48)
            st.dataframe(
                _h.style.format("{:.2f}", na_rep="-",
                                subset=_h.select_dtypes(include="float").columns.tolist())
            )

            st.subheader("Daily penalty summary")
            _d = daily.head(20)
            st.dataframe(
                _d.style.format("{:.2f}", na_rep="-",
                                subset=_d.select_dtypes(include="float").columns.tolist())
            )

            st.subheader("Monthly generation summary")
            st.dataframe(
                monthly_gen.style.format("{:.2f}", na_rep="-",
                                         subset=monthly_gen.select_dtypes(include="float").columns.tolist())
            )

            st.subheader("Monthly penalty summary")
            st.dataframe(
                monthly_pen.style.format("{:.2f}", na_rep="-",
                                         subset=monthly_pen.select_dtypes(include="float").columns.tolist())
            )

            st.subheader("Yearly summary")
            st.dataframe(
                yearly.style.format("{:.2f}", na_rep="-",
                                    subset=yearly.select_dtypes(include="float").columns.tolist())
            )

            out_buf = io.BytesIO()
            with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
                hourly.to_excel(writer, sheet_name="Hourly", index=False)
                daily.to_excel(writer, sheet_name="Daily penalty", index=False)
                monthly_gen.to_excel(
                    writer, sheet_name="Monthly generation", index=False
                )
                monthly_pen.to_excel(
                    writer, sheet_name="Monthly penalty", index=False
                )
                yearly.to_excel(writer, sheet_name="Yearly", index=False)

                # ---- extra sheets, only when the optimiser was used ----
                if optimizer_diag is not None:
                    pd.DataFrame(
                        [{k: v for k, v in optimizer_diag.items()}]
                    ).to_excel(writer, sheet_name="Optimiser diagnostics", index=False)

                    pd.DataFrame(
                        _optimizer_settings_rows(optimizer_params, params)
                    ).to_excel(writer, sheet_name="Optimiser settings", index=False)

                if baseline_hourly is not None:
                    cmp_export = pd.DataFrame(
                        [
                            {
                                "KPI": k,
                                "Rule-based": k_base.get(k, float("nan")),
                                "Optimised": k_opt[k],
                                "Delta": k_opt[k] - k_base.get(k, float("nan")),
                            }
                            for k in k_opt
                        ]
                    )
                    cmp_export.to_excel(writer, sheet_name="Engine comparison", index=False)
                    baseline_hourly.to_excel(
                        writer, sheet_name="Hourly (rule-based)", index=False
                    )
            out_buf.seek(0)

            safe_project = project_name.strip().replace(" ", "_")
            if not safe_project:
                safe_project = "Project"

            if sys.platform == "emscripten":
                st.caption(
                    "Excel download is disabled in the in-browser build. "
                    "All result tables are shown above."
                )
            else:
                st.download_button(
                    label="Download results as Excel",
                    data=out_buf,
                    file_name=f"{safe_project}_bess_results.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                )
            
 
        except Exception as e:
            st.error(f"Error during simulation: {e}")
            import traceback

            with st.expander("Traceback", expanded=False):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
