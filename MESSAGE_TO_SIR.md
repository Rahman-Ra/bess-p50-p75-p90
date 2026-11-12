# Message to Sir — completion report

**Subject: BESS P50/P75/P90 app — all nine review findings fixed and verified**

## Live app

**https://rahman-ra.github.io/bess-p50-p75-p90/**

Public, no login, and independent of my laptop. First load takes a few minutes
while the browser downloads Python; later loads are cached.

## What was delivered

The updated app is **`bess_app_v6.py`**. I left **`bess_app_v5.py`** unchanged
as the audited baseline, so every line number in your review still points at
the exact code you reviewed.

All eight findings in `rahman_bess_flaws.xlsx` and the verbal MILP look-ahead
comment are fixed. Each fix has its own check; all nine pass.

### 1. Degradation profile indexed by project year

The profile is 1..N but was read with the calendar year, so
`degr.get(2024)` always missed and degradation was silently ignored.

Both multi-year drivers now compute a 1-based project-year ordinal and use
that. A three-year frame labelled 2030–2032 applies profile entries 1, 2 and 3
exactly. **Check F1: PASS.**

### 2. Year-dependent RTE had the same bug

The RTE profile now uses the same project-year ordinal in both drivers.
Verified: efficiency follows entries 1 → 2 → 3 instead of falling back to
year 1. **Check F2: PASS.**

### 3. `n_years` did nothing

`expand_to_n_years` now makes the simulated series exactly `n_years` long,
repeating or truncating the input. A repeated non-leap year drops 29 February.
Verified: 1→3 repeats, 3→1 truncates, 3→5 cycles, exact case unchanged.
**Check F3: PASS.**

### 4. Initial SoC was 100 %

`BESSParams.initial_soc_pct` was added, defaulting to **50 % of the usable
band**. All four engines (rule single, rule split, MILP single, MILP split)
start there, clamp inside the min/max band, and the “reset each year” option
resets to the same value. The sidebar exposes it and the settings sheet
records it. Verified: 100 % → 45.0 MWh and 50 % → 22.5 MWh on a 45 MWh usable
band, in all four engines. **Check F4: PASS.**

### 5. Joint annual factor applied to each technology

Annual conversion is now **per technology** in both the app and
`quantile_scenarios.py`:

```text
solar_P_L = factor_solar,L × solar_P50
wind_P_L  = factor_wind,L  × wind_P50
total_P_L = solar_P_L + wind_P_L
```

Solar uses 0.9878 / 0.9736, wind uses 0.9696 / 0.9147. The joint factor
(0.9622 / 0.9500) is retained only as a portfolio diagnostic and is never
applied to either series. Summing per-technology P75s is conservative relative
to a true joint P75 when the resources are imperfectly correlated.
**Check F5: PASS.** The annual sample files and results were regenerated.

### 6. `solar_contract_share` was dead code

Implemented a real per-technology contract split end to end: rule split engine,
MILP split model with per-technology peak caps and penalties, both settlement
writers, a sidebar control, and the settings sheet. Verified for shares 1.00,
0.84, 0.50 and 0.00; solar-only and wind-only match their single-technology
equivalents. **Check F6: PASS.**

### 7. No chronological sort after load

`validate_input_frame` now performs a stable sort by Year, Month, Day, Hour
after coercion. Every engine goes through it, so an unsorted export can no
longer dispatch solar against the wrong hour. Verified with a deliberately
scrambled frame: it is reordered and the solar column realigns with its
timestamp. **Check F7: PASS.**

### 8. `fallback_windows` diagnostics overwritten

Numeric diagnostics are now summed across windows; only string fields keep
last-value semantics. Verified: a 9-window run reports 9 windows, 6 fallback
windows, 3 degenerate windows, 12 MILP rescues and 1.5 s solve time.
**Check F8: PASS.**

### 9. MILP look-ahead bias

The MILP is now a true receding-horizon controller:

- **Causal forecast:** seasonal-naive, value from 24 hours earlier, built only
  from data strictly before the window. No future values enter the model.
- **Receding horizon:** solve `[t, t+24 h)`, commit only hour `t`, re-solve
  next hour.
- **Physical settlement:** each committed hour is settled against realised
  generation, grid limit and state of charge. Forecast error cannot create
  fictitious export.
- **Honest benchmark:** the old perfect-foresight mode is retained but clearly
  labelled as an upper bound in the UI and the settings sheet.
- **Peak charging:** still forbidden by default; if explicitly allowed, the
  settlement executes the plan instead of silently zeroing it.

Verified on 720 h with zero fallbacks:

| Controller | Penalty (MWh) | Supplied (MWh) |
|---|---:|---:|
| Rule-based | 5,338.2 | 17,461.8 |
| MILP, causal receding horizon | 6,148.8 | 17,579.6 |
| MILP, perfect-foresight benchmark | 5,344.0 | 17,524.0 |

The causal MILP is slightly worse than the rule on this window. That is the
honest consequence of removing look-ahead: the rule is tuned for this plant and
the optimiser can no longer see future generation. The oracle result is close
to the rule, which confirms the optimiser is sound; the gap between causal and
oracle is the value of the forecast. **Checks F9 and F9 peak-charge: PASS.**

## Updated results (rule engine, 2024)

These use the corrected initial SoC (50 % of the usable band) and the
per-technology annual factors.

| Scenario | Penalty (MWh) | Supplied (MWh) | Delivery % |
|---|---:|---:|---:|
| Annual P50 | 45,299.2 | 262,996.7 | 94.55 |
| Annual P75 | 48,211.2 | 257,502.3 | 94.50 |
| Annual P90 | 53,081.5 | 248,745.4 | 94.41 |
| Hourly P50 | 45,299.2 | 262,996.7 | 94.55 |
| Hourly P75 | 99,851.0 | 190,065.1 | 94.02 |
| Hourly P90 | 137,051.4 | 146,357.3 | 93.77 |

The annual P90 adds about 7,800 MWh of penalty over P50, not a collapse. The
hourly P90 is much harsher (about 137,000 MWh) because it applies bad weather
in every hour; it is an operational stress test, not an annual bankable case.

## Verification summary

```text
F1/F2  PASS   F3  PASS   F4  PASS   F5  PASS   F6  PASS
F7     PASS   F8  PASS   F9  PASS   F9 peak-charge  PASS
UI integration (rule, causal MILP, split block, oracle, settings)  PASS
python -m pytest -q test_quantile_scenarios.py                     10 passed
```

`FLAW_FIXES.md` contains the finding-by-finding detail and the evidence above.

## How to run

```powershell
pip install -r requirements.txt
python -m streamlit run bess_app_v6.py
```

Upload `sample_data/battery_input_annual_p50.csv` (or the hourly file), choose
P50 / P75 / P90 in the **Generation scenario** selector, and click **Run BESS
dispatch**. One upload contains all three scenarios.

The MILP defaults to the causal information set. For the benchmark, choose
**Realised generation (perfect-foresight benchmark)** in the sidebar; the app
labels it as an upper bound.

## Caveats

- The sample data are illustrative and are not a plant recommendation.
- Annual P90 needs 10+ years of data to be stable.
- The absolute KPIs should not be read as a final design. Substitute the actual
  capacities, contract and battery, then compare the P50/P75/P90 deltas.
- The perfect-foresight MILP is a benchmark, not an operational controller.
  Use the causal mode for deployable results.

## Package

The package contains `bess_app_v6.py`, `bess_app_v5.py` (audited baseline),
`quantile_scenarios.py`, the experiment runners, the sample inputs, the updated
result tables and figures, the fitted calibration, `FLAW_FIXES.md`,
`SCENARIO_GUIDE.md` and this report.
