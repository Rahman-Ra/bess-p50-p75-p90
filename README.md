# BESS Wind–Solar P50 / P75 / P90 Package

This package contains the updated BESS dispatch app and the P50 → P75/P90
scenario analysis prepared for review.

## What was requested

1. Take the **P50 wind forecast** and **P50 solar forecast** and feed them into
   the battery model as generation inputs.
2. Add conversion logic to change those P50 inputs into **P75** and **P90**.
3. Explain the method and show how to implement it.

## What is included

| File | Purpose |
|---|---|
| `bess_app_v4_fixed.py` | Streamlit BESS app with a **Generation scenario** selector (P50 / P75 / P90). |
| `quantile_scenarios.py` | Fits and applies the P50 → P75/P90 conversion; writes battery-ready input files. |
| `run_scenario_experiment.py` | Runs the rule/MILP battery engine on P50, P75 and P90 and compares KPIs. |
| `run_experiment.py` | Shared experiment harness used by `run_scenario_experiment.py`. |
| `sample_data/battery_input_annual_p50.csv` | Ready-to-upload annual scenario file. It contains all three scenarios (P50/P75/P90). |
| `sample_data/battery_input_annual_p75.csv` | Annual P75 dispatch input (`solar_MWh`/`wind_MWh` already set to P75). |
| `sample_data/battery_input_annual_p90.csv` | Annual P90 dispatch input. |
| `sample_data/battery_input_hourly_p50.csv` | Ready-to-upload hourly stress-test scenario file. It contains all three scenarios. |
| `sample_data/battery_input_hourly_p75.csv` | Hourly P75 dispatch input. |
| `sample_data/battery_input_hourly_p90.csv` | Hourly P90 dispatch input. |
| `results/scenario_kpis_annual.md` | Annual P50/P75/P90 battery comparison. |
| `results/scenario_kpis_hourly.md` | Hourly P50/P75/P90 battery comparison. |
| `results/penalty_scenarios.png` | Penalty chart by scenario. |
| `results/scenario_week.png` | Example week of solar/wind scenarios. |
| `calibration/` | Fitted calibration factors and offsets for audit. |
| `SCENARIO_GUIDE.md` | Full method, formulas, commands and caveats. |
| `MESSAGE_TO_SIR.md` | Draft message to send with this package. |
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
   python -m streamlit run bess_app_v4_fixed.py
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

7. A new **Generation scenario** selector appears. Choose **P50**, **P75** or
   **P90**, then click **Run BESS dispatch**.

The sample file contains all three scenarios, so the same upload can be run
three times without uploading another file.

### Reproduce the battery comparison from the command line

The annual and hourly P75/P90 battery inputs are included, so no regeneration is
needed:

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
`--engine milp` for the rolling-horizon optimiser.

## How the conversion works

### P90 convention

P90 is the **90% probability-of-exceedance** value; for generation it is the
conservative low case:

```text
P90 <= P75 <= P50
```

It is the 10th percentile of the generation distribution, not the 90th
percentile.

### Hourly mode (operational stress test)

```text
residual(t) = actual(t) - P50(t)
P_L(t) = max(0, P50(t) + Q_(1-L/100)(residual | month, hour))
```

Residual quantiles are fitted by month × hour and shrunk toward the global
quantile where a group has few observations. An additive correction is used
because solar has many zero/near-zero hours, where a multiplicative correction
would collapse to zero.

### Annual mode (bankable / firm-capacity case)

```text
factor_L = Q_(1-L/100)(annual energy) / median(annual energy)
P_L(t) = factor_L × P50(t)
```

The annual factors are fitted on 10 years (2015–2024) of annual energy and
applied to the hourly P50 shape.

### Shared solar + wind battery

For a shared BESS, solar and wind are calibrated **jointly**:

```text
total_P_L = calibration(P50_solar + P50_wind)
solar_P_L = total_P_L × P50_solar / (P50_solar + P50_wind)
wind_P_L  = total_P_L - solar_P_L
```

This avoids adding solar-P90 and wind-P90, which would double-count
uncertainty.

## Results (rule engine, 2024)

| Case | Penalty (MWh) | Supplied (MWh) |
|---|---:|---:|
| Annual P50 | 45,140.5 | 263,263.3 |
| Annual P75 | 50,608.6 | 253,074.3 |
| Annual P90 | 52,457.8 | 249,806.4 |
| Hourly P50 | 45,140.5 | 263,263.3 |
| Hourly P75 | 99,642.0 | 190,328.0 |
| Hourly P90 | 136,813.6 | 146,618.0 |

Annual P90 factors (2015–2024):

| Resource | P75 | P90 |
|---|---:|---:|
| Solar | 0.988 | 0.974 |
| Wind | 0.970 | 0.915 |
| Combined solar+wind | 0.962 | 0.950 |

Use **annual P90** for bankability, firm capacity, revenue and debt sizing.
Use **hourly P90** for operational stress testing and hour-by-hour reliability.

## Verification

```text
python -m pytest -q test_quantile_scenarios.py
# 8 passed
```

The app compiles and imports cleanly, and the scenario tests above cover the
calibration, joint allocation and battery-input writer.

## Notes

- The sample data are illustrative and are not a plant recommendation.
- Annual P90 needs 10+ years of data to be stable.
- The absolute KPI values should not be read as a final plant design; compare
  the deltas between P50, P75 and P90 after substituting the actual capacities,
  contract and battery.
