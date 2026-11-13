# P50 / P75 / P90 for the wind–solar–BESS model

This note explains what the P-values mean, why they cannot be produced by a
fixed haircut, and how to run the implementation in this repository.

The short version:

- **P50** is the central case (median).
- **P75** is the value exceeded in 75 % of cases.
- **P90** is the value exceeded in 90 % of cases, so it is the **conservative
  low** case, not the 90th percentile.
- For generation: **P90 ≤ P75 ≤ P50**.
- There are two different P90s:
  - **annual P90** — the standard finance/resource number;
  - **hourly P90** — the 10th percentile of each hour's predictive
    distribution, much more severe.
- **Annual P-values are per technology.** Solar and wind have very different
  interannual variability, so each uses its own annual factor and the total is
  their sum. The joint solar+wind factor is a portfolio diagnostic only; it is
  never applied to either resource.
- **Hourly P-values pool the joint total.** The hourly residual quantile is
  fitted on the combined solar+wind series and split back pro rata on the P50
  shares, because the resources share one battery and their hourly errors are
  correlated.

## 1. What the LinkedIn post implies for the battery

The post shows that India's 2026 El Niño year had a large hydro shortfall
while solar and wind grew. The resilience argument is that a diversified
renewable portfolio plus storage can cover the evening peak. For a BESS model,
that translates into a firmness question:

- How much energy can the battery be expected to deliver in a **bad** renewable
  year, not just an average one?
- How much peak shortfall penalty appears under P75/P90?
- Does the battery need to be larger, or is the constraint transmission/
  contract rather than energy?

A P50-only dispatch answers "what happens in an average year". A P90 dispatch
answers "what happens in the year the lender/operator must survive". Both are
useful; they answer different questions.

## 2. Why "P90 = 0.8 × P50" is wrong

A fixed factor is not a probability statement. It over-penalises stable hours
and under-protects uncertain ones. On the out-of-sample LightGBM predictions in
`oos_out/` (calibration 2020–2022, test 2023–2024):

| Case | Hourly coverage of actual ≥ scenario |
|---|---|
| fixed `0.8 × P50` | ~45 % |
| empirical P90 | ~89–90 % |

So the fixed haircut is not P90 at all. The empirical residual quantile is the
minimum honest conversion. The implementation uses

```
P_L(t) = P50(t) + q_(1-L/100)( actual - P50 | month, hour )
```

for the hourly mode. The residual quantile is estimated globally and inside
month × hour groups, with small groups shrunk towards the global value. The
offsets are in MWh and are written to `scenario_calibration.json`, so every
number is auditable.

## 3. Annual P90 versus hourly P90

These are not the same quantity, and mixing them is the most common mistake.

Using the 10 years in `real_input.csv` (2015–2024), the annual-energy factors
relative to the median year are:

| Resource | P75 factor | P90 factor | Annual CV |
|---|---:|---:|---:|
| Solar | 0.9878 | 0.9736 | 2.8 % |
| Wind | 0.9696 | 0.9147 | 7.9 % |
| Joint diagnostic | 0.9622 | 0.9500 | 3.9 % |

Wind is much more variable year-to-year than solar; the combined portfolio is
smoother than either resource alone. This is the diversification point in the
post.

The factors are applied **per technology**:

```text
solar_P_L(t) = 0.9878 or 0.9736 × solar_P50(t)
wind_P_L(t)  = 0.9696 or 0.9147 × wind_P50(t)
total_P_L(t) = solar_P_L(t) + wind_P_L(t)
```

The joint factor is shown only to quantify diversification. Applying it to
both resources, as the previous version did, overstated solar risk and
understated wind risk; that was review finding 5.

Summing per-technology P75s is conservative relative to a true joint P75 when
the resources are imperfectly correlated, so it does not double-count
uncertainty in the conservative direction. If a lender asks for a portfolio
P75 rather than a resource-level P75, use the joint diagnostic explicitly and
state the correlation assumption.

The **hourly** P90 is far more severe. In the same test window, wind P90 has a
mean of about 4.3 MWh against an actual mean of 14.7 MWh. That is a legitimate
hourly risk statement, but it is not an annual energy P90. If you feed hourly
P90 into every hour of a year, the battery sees a resource that is dramatically
worse than any realistic annual P90 year.

**Use annual P90 for:** firm capacity, revenue, debt sizing, annual KPI
reporting, "bankable" cases.

**Use hourly P90 for:** real-time reserve, hour-by-hour reliability, stress
testing the dispatch controller, checking whether the battery can cover a
specific evening peak.

## 4. What was added

| File | Purpose |
|---|---|
| `quantile_scenarios.py` | Fit P75/P90 from historical forecast errors or annual energy; write hourly scenarios and battery-ready inputs. |
| `run_scenario_experiment.py` | Run the existing rule/MILP dispatch on P50/P75/P90 inputs and compare KPIs. |
| `run_experiment.py` | Shared harness for causal-vs-oracle MILP experiments. |
| `test_quantile_scenarios.py` | Tests for calibration, monotonicity, allocation and file writing. |
| `bess_app_v6.py` | Current single-file app: P50/P75/P90 selector, embedded calibration, all review fixes plus the independent re-audit corrections. |
| `bess_app_v5.py` | Audited baseline from the previous review; kept unchanged for line-number traceability. |
| `FLAW_FIXES.md` | Finding-by-finding fix record and verification evidence. |

## 5. Commands

The commands below use the full project's raw forecast files (`oos_out/` and
`real_input.csv`) and are included for provenance. The package itself ships the
ready-made P75/P90 battery inputs, so the comparison can be run directly with
the commands in `README.md` — no regeneration step is required.

### 5.1 Hourly P-values (operational stress test)

```powershell
python quantile_scenarios.py `
  --solar-pred oos_out/lgbm_solar_pred.csv `
  --wind-pred  oos_out/lgbm_wind_pred.csv `
  --train-years 2020,2021,2022 --test-years 2023,2024 `
  --mode hourly --group month,hour --aggregation joint --calibrate-p50 `
  --base-input real_input.csv --apply-year 2024 `
  --outdir scenario_hourly
```

Outputs:

- `scenario_hourly.csv` — time, actual, P50/P75/P90 for solar, wind, total;
- `scenario_summary.csv` — hourly coverage, bias and pinball loss;
- `annual_summary.csv` — annual energy by scenario;
- `battery_input_p50.csv`, `battery_input_p75.csv`, `battery_input_p90.csv` —
  full BESSTOOL inputs ready for the app or the experiment runner.

### 5.2 Annual P-values (bankable case)

```powershell
python quantile_scenarios.py `
  --solar-pred oos_out/lgbm_solar_pred.csv `
  --wind-pred  oos_out/lgbm_wind_pred.csv `
  --train-years 2020,2021,2022 --test-years 2023,2024 `
  --mode annual --aggregation joint `
  --annual-actual real_input.csv `
  --base-input real_input.csv --apply-year 2024 `
  --outdir scenario_annual
```

`--annual-actual` is important when the forecast file only covers the test
period. It fits the P-values on the full 10-year resource record while still
derating the P50 forecast series.

Annual conversion is per technology regardless of `--aggregation`: the joint
factor printed in this mode is a portfolio diagnostic, never an input to the
solar or wind series.

### 5.3 Battery comparison

```powershell
python run_scenario_experiment.py `
  --inputs scenario_annual/battery_input_p50.csv,scenario_annual/battery_input_p75.csv,scenario_annual/battery_input_p90.csv `
  --labels P50,P75,P90 --engine rule `
  --outdir scenario_dispatch_annual
```

Use `--engine milp` for the receding-horizon optimiser. Start with `--hours 720`
if you want a one-month check before running a full year. The MILP defaults to
the causal information set; add `--forecast-mode oracle` for the explicitly
labelled perfect-foresight benchmark.

**Reproducing the shipped numbers.** `scenario_hourly_raw/` is the uncalibrated
P50 variant (`calibrate_p50: false`); the §5.1 command uses `--calibrate-p50`
and therefore changes the P50 case. The P75/P90 columns are identical. For the
documented KPI table, use the battery inputs in `sample_data/`, which are the
exact files that produced `results/scenario_kpis_annual.md` and
`results/scenario_kpis_hourly.md`.

**Joint-mode column names (hourly only).** With `--aggregation joint` in hourly
mode, `solar_pXX_MWh` and `wind_pXX_MWh` are proportional allocations of the
joint solar+wind scenario. They are not standalone resource PXX values; the
scenario level is `total_pXX_MWh`. Annual mode is always per technology: the
columns are true resource-level PXX values and the total is their sum.

## 6. Using the Streamlit app

Start `bess_app_v6.py`. Upload any `battery_input_p*.csv` file. If the file
contains `solar_p50_MWh` / `solar_p75_MWh` / `solar_p90_MWh` (and the wind
equivalents), a **Generation scenario** selector appears in the sidebar. The
app replaces `solar_MWh` and `wind_MWh` with the chosen scenario before
dispatch, leaving the original columns in the file for audit.

If the file has only `solar_MWh` and `wind_MWh`, the app offers the embedded
**Scenario conversion** (annual factor or hourly residual) and then the
**Generation scenario** selector.

Additional controls that matter for the review fixes:

- **Initial state of charge (% of usable band)** — default 50 %, shared by all
  engines;
- **Number of years (repeat same TS)** — defines the exact simulated horizon;
- **Information set** — *Causal forecast* (default, deployable) or *Realised
  generation* (perfect-foresight benchmark, explicitly labelled as an upper
  bound);
- **Receding horizon** — solve 24 h ahead and commit 1 h;
- **Solar contract share** — per-technology split in split mode.

## 7. What the demo run shows

Rule-based engine, 2024, illustrative plant parameters (281 MW solar,
190 MW contract, 600 MWh / 200 MW BESS):

| Scenario | Penalty (MWh) | Supplied (MWh) | BESS export (MWh) | Delivery % |
|---|---:|---:|---:|---:|
| Annual P50 | 45,299 | 262,997 | 208,120 | 94.55 |
| Annual P75 | 48,211 | 257,502 | 205,930 | 94.50 |
| Annual P90 | 53,081 | 248,745 | 202,349 | 94.41 |
| Hourly P50 | 45,299 | 262,997 | 208,120 | 94.55 |
| Hourly P75 | 99,851 | 190,065 | 166,943 | 94.02 |
| Hourly P90 | 137,051 | 146,357 | 135,147 | 93.77 |

The annual P90 adds about 7,800 MWh of penalty over P50, not a collapse. The
hourly P90 is much harsher (about 137,000 MWh penalty) because it assumes bad
weather in every hour. That difference is the annual-versus-hourly distinction
in numbers. These values use the corrected initial SoC (50 % of the usable
band) and the per-technology annual factors.

The absolute values are not a plant recommendation: the sample data are
per-100-MW resource profiles and the contract/BESS settings are the paper
defaults. Compare the **deltas** between scenarios, then substitute your own
capacities, contract and battery.

## 8. How to use the calibration in your own code

```python
import pandas as pd
import quantile_scenarios as qs

train = pd.read_csv("solar_train.csv", parse_dates=["time"])   # time, actual, pred
cal = qs.fit_exceedance_calibration(
    train["actual"], train["pred"],
    levels=(50, 75, 90), group_cols=("month", "hour"),
    calibrate_p50=True,
    month=train["time"].dt.month, hour=train["time"].dt.hour,
)

test = pd.read_csv("solar_test.csv", parse_dates=["time"])
scen = cal.scenario(
    test["pred"],
    month=test["time"].dt.month, hour=test["time"].dt.hour,
)
test["p50"] = scen[50]
test["p75"] = scen[75]
test["p90"] = scen[90]
```

For annual factors, apply each resource's own factor:

```python
solar_cal = qs.fit_annual_factors(solar_annual, years, levels=(50, 75, 90))
wind_cal = qs.fit_annual_factors(wind_annual, years, levels=(50, 75, 90))
s_scen = solar_cal.scenario(solar_p50_hourly)   # scalar factor, shape preserved
w_scen = wind_cal.scenario(wind_p50_hourly)
total_p75 = s_scen[75] + w_scen[75]
total_p90 = s_scen[90] + w_scen[90]
```

Do not multiply both resources by the joint factor; that is review finding 5.

## 9. Caveats and next steps

1. **P50 must be a real median.** The LightGBM point forecast in this repository
   is an RMSE-optimal forecast, not a median; it predicts a small positive
   value at night when actual solar is zero. `--calibrate-p50` fixes that. If
   your P50 already comes from a calibrated resource model, leave the flag off.
2. **Annual P-values need 10+ years.** Three years give a noisy P90. Use
   `--annual-actual` with the full record.
3. **Annual per technology, hourly joint.** Annual factors are resource
   statements: apply each resource's own factor and sum the series. Hourly
   residual quantiles are fit on the joint total and split pro rata, because
   the resources share one battery and their hourly errors are correlated.
   The joint annual factor is a diagnostic, not an input to either series.
4. **Deterministic scenarios are not a robust optimisation.** Running P50, P75
   and P90 separately shows sensitivity. A truly firm dispatch would optimise
   against a weighted scenario set (e.g. 0.5/0.3/0.2) or a min-max regret
   formulation. That is the logical next step if the P90 penalty is
   unacceptable.
5. **Hydro is missing.** The post's shock was hydro. If hydro is part of the
   contract, add a hydro P-value column and include it in the joint calibration;
   otherwise the model still assumes hydro is firm.
6. **Correlation matters.** The joint calibration pools solar and wind errors.
   If you have many years and locations, fit a copula or bootstrap whole days
   instead of pooling hourly residuals; that preserves weather persistence.
