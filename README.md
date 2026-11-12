# BESS Wind–Solar P50 / P75 / P90 Package (v6)

This package contains the BESS dispatch app, the P50 → P75/P90 scenario
analysis, and the fixes for the nine review findings. The current app is
**`bess_app_v6.py`**. The previously reviewed `bess_app_v5.py` is kept
unchanged as the audited baseline.

## Live app (public, no login)

**https://rahman-ra.github.io/bess-p50-p75-p90/**

Anyone can open this link; it does not depend on your laptop. It runs the
Streamlit UI in the browser through GitHub Pages and Stlite. The first visit
takes a few minutes while Python and the libraries download; later visits are
cached. This build uses the **rule-based engine** (the engine behind the
published results) and keeps the result tables on screen; the Excel download is
disabled in the browser build. The full Streamlit deployment adds the MILP
engine and Excel download.

## What was requested

1. Take the **P50 wind forecast** and **P50 solar forecast** and feed them into
   the battery model as generation inputs.
2. Add conversion logic to change those P50 inputs into **P75** and **P90**.
3. Explain the method and show how to implement it.
4. Fix the nine issues returned in review.

## The nine review findings

All nine are fixed in `bess_app_v6.py`. The full evidence is in
[`FLAW_FIXES.md`](FLAW_FIXES.md).

| # | Finding | Fix |
|---|---|---|
| 1 | Degradation profile indexed 1..N but looked up by calendar year | Project-year ordinal used in both multi-year drivers |
| 2 | Same bug for year-dependent RTE | Same ordinal used for RTE |
| 3 | `n_years` control unused | Input is repeated/truncated to exactly `n_years` |
| 4 | Initial SoC = 100 % | `initial_soc_pct` default 50 % of usable band, all four engines |
| 5 | Joint annual factor applied to each technology | Per-technology annual factors; joint kept as diagnostic |
| 6 | `solar_contract_share` hardcoded at 84/16 | Real per-technology contract split in both split engines + UI |
| 7 | No chronological sort after load | Stable sort in `validate_input_frame`, used by every path |
| 8 | `fallback_windows` diagnostics overwritten | Numeric diagnostics aggregated across windows |
| 9 | MILP had look-ahead | Causal 24-h-lag forecast, true receding horizon, physical settlement |

## What is included

| File | Purpose |
|---|---|
| `bess_app_v6.py` | **Use this one.** Single file: Streamlit BESS app with the P50/P75/P90 selector, embedded calibration, and all nine fixes. |
| `bess_app_v5.py` | Audited baseline from the previous review. Kept unchanged so the review line numbers stay valid; it does **not** contain the nine fixes. |
| `bess_app_v4_fixed.py` | Older app without the embedded calibration: reads P75/P90 columns that already exist in the upload. |
| `quantile_scenarios.py` | Fits and applies the P50 → P75/P90 conversion; writes battery-ready input files. |
| `run_scenario_experiment.py` | Runs the rule/MILP battery engine on P50, P75 and P90 and compares KPIs. |
| `run_experiment.py` | Shared experiment harness, including causal-vs-oracle MILP runs. |
| `sample_data/battery_input_annual_p50.csv` | Ready-to-upload annual scenario file with all three scenarios. |
| `sample_data/battery_input_annual_p75.csv` | Annual P75 dispatch input (`solar_MWh`/`wind_MWh` already set to P75). |
| `sample_data/battery_input_annual_p90.csv` | Annual P90 dispatch input. |
| `sample_data/battery_input_hourly_p50.csv` | Ready-to-upload hourly stress-test scenario file with all three scenarios. |
| `sample_data/battery_input_hourly_p75.csv` | Hourly P75 dispatch input. |
| `sample_data/battery_input_hourly_p90.csv` | Hourly P90 dispatch input. |
| `results/scenario_kpis_annual.md` | Annual P50/P75/P90 battery comparison. |
| `results/scenario_kpis_hourly.md` | Hourly P50/P75/P90 battery comparison. |
| `results/penalty_scenarios.png` | Penalty chart by scenario. |
| `results/scenario_week.png` | Example week of solar/wind scenarios. |
| `calibration/` | Fitted calibration factors and offsets for audit. |
| `FLAW_FIXES.md` | Finding-by-finding fix description and verification evidence. |
| `SCENARIO_GUIDE.md` | Full method, formulas, commands and caveats. |
| `MESSAGE_TO_SIR.md` | Completion report for the reviewer. |
| `DEPLOYMENT.md` | How to publish the Streamlit UI for anyone to view. |

## Quick start (Windows)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   python -m streamlit run bess_app_v6.py
   ```

   Or double-click `RUN-APP.bat`.

5. Open the URL Streamlit prints, normally:

   ```text
   http://localhost:8501
   ```

6. In the sidebar, use **Upload BESSTOOL_INPUT** and upload:

   ```text
   sample_data/battery_input_annual_p50.csv
   ```

7. Choose **P50**, **P75** or **P90** in the **Generation scenario** selector,
   then click **Run BESS dispatch**.

The sample file contains all three scenarios, so the same upload can be run
three times without uploading another file.

### Reproduce the battery comparison from the command line

```powershell
# annual, 2024
python run_scenario_experiment.py `
  --inputs sample_data/battery_input_annual_p50.csv,sample_data/battery_input_annual_p75.csv,sample_data/battery_input_annual_p90.csv `
  --labels P50,P75,P90 --engine rule --outdir dispatch_annual

# hourly stress test
python run_scenario_experiment.py `
  --inputs sample_data/battery_input_hourly_p50.csv,sample_data/battery_input_hourly_p75.csv,sample_data/battery_input_hourly_p90.csv `
  --labels P50,P75,P90 --engine rule --outdir dispatch_hourly
```

Add `--hours 720` for a one-month check before a full-year run. Use
`--engine milp` for the receding-horizon optimiser. The MILP defaults to the
**causal** information set; add `--forecast oracle` for the perfect-foresight
benchmark.

## How the conversion works

### P90 convention

P90 is the **90 % probability-of-exceedance** value; for generation it is the
conservative low case:

```text
P90 <= P75 <= P50
```

It is the 10th percentile of the generation distribution, not the 90th
percentile.

### Hourly mode (operational stress test)

```text
residual(t) = actual(t) - P50(t)
total_P_L(t) = max(0, total_P50(t) + Q_(1-L/100)(residual | month, hour))
solar_P_L(t) = total_P_L(t) × solar_P50(t) / total_P50(t)
wind_P_L(t)  = total_P_L(t) - solar_P_L(t)
```

Residual quantiles are fitted by month × hour and shrunk toward the global
quantile where a group has few observations. An additive correction is used
because solar has many zero/near-zero hours, where a multiplicative correction
would collapse to zero.

### Annual mode (bankable / firm-capacity case)

Annual factors are applied **per technology**:

```text
solar_P_L(t) = factor_solar,L × solar_P50(t)
wind_P_L(t)  = factor_wind,L  × wind_P50(t)
total_P_L(t) = solar_P_L(t) + wind_P_L(t)
factor_X,L   = Q_(1-L/100)(annual energy of X) / median(annual energy of X)
```

The factors are fitted on 10 years (2015–2024) of annual energy and applied to
the hourly P50 shape. This is the correct treatment because solar and wind have
very different interannual variability: at P90 solar keeps 97.4 % of its
median year while wind keeps 91.5 %.

The joint solar+wind factor (0.962 P75 / 0.950 P90) is **not** applied to
either technology. It is reported only as a portfolio diagnostic: summing
per-technology P75s is conservative relative to a true joint P75 when the
resources are imperfectly correlated.

## Single-file app (`bess_app_v6.py`)

Drop it anywhere, install `streamlit pandas numpy openpyxl xlsxwriter`, then:

```text
streamlit run bess_app_v6.py
```

Upload a plain `BESSTOOL_INPUT` file — only `solar_MWh` and `wind_MWh` are
required. When the file carries no pre-built scenario columns, the sidebar adds:

- **Scenario conversion** — *Annual factor (bankability)* or *Hourly residual
  (stress test)*;
- **Generation scenario** — P50 / P75 / P90.

The fitted calibration is embedded (288 month × hour groups plus the annual
factors), so no second file is needed.

Controls added or corrected in v6:

- **Initial state of charge (% of usable band)** — default 50 %;
- **Number of years (repeat same TS)** — now actually defines the horizon;
- **Information set** — causal (default) or perfect-foresight benchmark;
- **Receding horizon** — solve 24 h ahead, commit 1 h;
- **Solar contract share** — visible in split mode.

## Results (rule engine, 2024)

| Case | Penalty (MWh) | Supplied (MWh) | BESS export (MWh) | Delivery % |
|---|---:|---:|---:|---:|
| Annual P50 | 45,299.2 | 262,996.7 | 208,120.1 | 94.55 |
| Annual P75 | 48,211.2 | 257,502.3 | 205,930.2 | 94.50 |
| Annual P90 | 53,081.5 | 248,745.4 | 202,348.6 | 94.41 |
| Hourly P50 | 45,299.2 | 262,996.7 | 208,120.1 | 94.55 |
| Hourly P75 | 99,851.0 | 190,065.1 | 166,943.1 | 94.02 |
| Hourly P90 | 137,051.4 | 146,357.3 | 135,146.5 | 93.77 |

These numbers use the corrected initial SoC (50 % of the usable band) and the
per-technology annual factors. They differ slightly from the previous package,
which started every run with a full battery.

Annual P90 factors (2015–2024):

| Resource | P75 | P90 |
|---|---:|---:|
| Solar | 0.9878 | 0.9736 |
| Wind | 0.9696 | 0.9147 |
| Joint diagnostic | 0.9622 | 0.9500 |

Use **annual P90** for bankability, firm capacity, revenue and debt sizing.
Use **hourly P90** for operational stress testing and hour-by-hour reliability.

### MILP information-set benchmark (720 h, 2024 P50)

| Controller | Penalty (MWh) | Supplied (MWh) |
|---|---:|---:|
| Rule-based | 5,338.2 | 17,461.8 |
| MILP, causal receding horizon | 6,148.8 | 17,579.6 |
| MILP, perfect-foresight benchmark | 5,344.0 | 17,524.0 |

The causal MILP is slightly worse than the rule on this window. That is the
honest consequence of removing look-ahead: the rule is tuned for this plant,
and the optimiser no longer sees future generation. The oracle result is close
to the rule, which confirms the optimiser is sound; the causal-to-oracle gap is
the value of the forecast.

## Verification

```text
python -m pytest -q test_quantile_scenarios.py
# 10 passed

F1/F2, F3, F4, F5, F6, F7, F8, F9, F9 peak-charge  all PASS
UI integration (rule, causal MILP, split block, oracle, settings)  PASS
```

The app compiles and imports cleanly. `FLAW_FIXES.md` lists what each check
proves.

## Notes

- The sample data are illustrative and are not a plant recommendation.
- Annual P90 needs 10+ years of data to be stable.
- The absolute KPI values should not be read as a final plant design; compare
  the deltas between P50, P75 and P90 after substituting the actual capacities,
  contract and battery.
- The perfect-foresight MILP is a benchmark, not an operational controller.
  Use the causal mode for deployable results.
