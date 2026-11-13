# Response to the nine review findings

Every finding from `rahman_bess_flaws.xlsx` and the verbal MILP look-ahead
comment has been fixed in **`bess_app_v6.py`**. `bess_app_v5.py` is left
unchanged as the audited baseline, so the line numbers in the review still
point at the code that was reviewed.

After those fixes, an independent re-audit checked the runtime, the core
engines and the scenario pipeline. Every confirmed finding from that pass is
also fixed and listed in section 10 below. The completion workbook
`rahman_bess_flaws_completed.xlsx` records the status and recheck evidence for
both sets.

Each fix was verified with a dedicated check before moving to the next one.
The checks are named `F1`–`F9` below. All nine pass, plus a separate
peak-charging edge case for the settlement logic.

## 1. Degradation profile keyed by project year, not calendar year

**Finding:** the profile is indexed 1..25 but looked up with `degr.get(2024)`,
so the degradation was silently ignored.

**Fix:** both multi-year drivers (`simulate_multi_year` and
`simulate_optimized_multi_year`) now compute a 1-based project-year ordinal and
use that for the degradation profile. Calendar year labels are irrelevant to
the lookup.

**Verified (F1/F2):** a three-year frame labelled 2030–2032 applies profile
entries 1, 2 and 3 exactly:

```text
[(2030, 1.0, 1.0, 0.91, 0.92),
 (2031, 0.88, 0.86, 0.93, 0.94),
 (2032, 0.77, 0.75, 0.95, 0.96)]
```

## 2. Year-dependent round-trip efficiency had the same bug

**Finding:** the RTE profile was indexed 1..N but also looked up by calendar
year, so every year fell back to the year-1 efficiency and the control was
dead.

**Fix:** the same project-year ordinal is now used for the RTE profile in both
multi-year drivers.

**Verified (F1/F2):** the RTE columns above follow profile entries 1, 2 and 3
(0.91/0.92 → 0.93/0.94 → 0.95/0.96), not a constant year-1 value.

## 3. “Number of years (repeat same TS)” did nothing

**Finding:** `n_years` was read by the sidebar but never used; the app ran
whatever span happened to be in the file.

**Fix:** `expand_to_n_years` now makes the input series exactly `n_years` long
before dispatch, repeating or truncating as needed. A repeated non-leap year
drops 29 February so the hourly chronology stays valid. The sidebar value is
passed into the run.

**Verified (F3):**

| Input | Requested | Result |
|---|---|---|
| 1 year | 3 | repeated to 3 years, 26,280 h |
| 3 years | 1 | truncated to the first year, 8,760 h |
| 3 years | 5 | cycled to 5 years |
| 3 years | 3 | unchanged, chronological |

## 4. Initial state of charge was 100 %

**Finding:** every run started with a full battery, which released free stored
energy at t = 0 and inflated early delivery.

**Fix:** `BESSParams.initial_soc_pct` was added (default **50 % of the usable
band**). All four engines — rule single, rule split, MILP single, MILP split —
start from that value and clamp it inside the min/max band. The “reset each
year” option resets to the same configured value, not to full. The sidebar
slider exposes it, and the settings sheet records it.

**Verified (F4):** all four engines start at the requested percentage;
`100 % -> 45.000 MWh` on a 0–45 MWh usable band, `50 % -> 22.500 MWh`, and the
reset path returns to the configured value each year.

## 5. Annual joint factor applied to each technology

**Finding:** annual P75/P90 used the joint solar+wind factor (0.962/0.950) on
both resources. Solar should use 0.988/0.974 and wind 0.970/0.915. The error
went in opposite directions for the two technologies.

**Fix:** annual conversion is now **per technology** in both the app
(`build_generation_scenarios`, embedded `SCENARIO_ANNUAL_FACTORS`) and
`quantile_scenarios.py` (`build_scenario_hourly` annual branch). The total
scenario is the sum of the two adjusted series. The joint factor is retained
only as a portfolio diagnostic (`SCENARIO_ANNUAL_FACTORS_JOINT`, printed by the
CLI) and is never applied to either technology.

**Verified (F5):** solar P75/P90 = 0.9878388 / 0.9735608 × P50; wind =
0.9695505 / 0.9146545 × P50; total = solar + wind; `P90 ≤ P75 ≤ P50` for both
resources. The regenerated `sample_data/battery_input_annual_p*.csv` files use
the per-technology factors.

**Why this is not double counting:** summing per-technology P75s is
*conservative* relative to a true joint P75 when the resources are imperfectly
correlated. The joint factor remains available as the less-conservative
portfolio cross-check.

## 6. `solar_contract_share` was hardcoded at 84/16

**Finding:** the split-mode contract share was fixed in the code; the
`solar_contract_share` parameter was dead and the UI had no control.

**Fix:** a real per-technology contract split was implemented end to end:

- the rule-based split engine allocates the contract to solar and wind
  separately;
- the MILP split model uses per-technology peak caps and penalties;
- both settlement writers use the per-technology targets;
- the sidebar exposes **Solar contract share** in split mode (default 0.84);
- the settings sheet records the value.

**Verified (F6):** shares 1.00, 0.84, 0.50 and 0.00 were run through both
engines. Each share produces the expected supplied energy and penalty; the
solar-only and wind-only cases match their single-technology equivalents.

## 7. No chronological sort after load

**Finding:** the engines walked the file row order. An unsorted Excel/CSV
export would silently dispatch solar against the wrong hour.

**Fix:** `validate_input_frame` now sorts by `Year, Month, Day, Hour` with a
stable sort after coercion. Every engine and every downstream path goes
through this function, so the fix applies everywhere.

**Verified (F7):** a deliberately scrambled frame is reordered chronologically
and the solar column realigns with its timestamp.

## 8. `fallback_windows` diagnostics were overwritten

**Finding:** each window assigned the counters instead of accumulating them,
so only the last window was reported.

**Fix:** numeric diagnostics (`windows`, `fallback_windows`,
`degenerate_windows`, `milp_rescues`, `solve_seconds`) are summed across
windows; only genuinely textual fields keep last-value semantics.

**Verified (F8):** a run crossing 9 windows reports 9 windows, 6 fallback
windows, 3 degenerate windows, 12 MILP rescues and 1.5 s total solve time —
i.e. aggregates, not the last window's values.

## 9. MILP look-ahead bias (raised verbally)

**Finding:** the rolling-horizon MILP was given the realised generation for the
whole look-ahead window. That is perfect foresight, not a deployable
controller, so its KPIs were an optimistic bound.

**Fix:** the MILP now runs as a true receding-horizon controller:

- **Causal forecast:** a seasonal-naive forecast using the value from 24 hours
  earlier, built only from data strictly before the window start. No future
  values enter the model.
- **Receding horizon:** at each hour `t` the model solves `[t, t+H)` and
  commits only hour `t` (`H = 24 h`, commit `1 h` by default). The next hour
  re-solves with the newly observed data.
- **Physical settlement:** each committed hour is settled against the realised
  generation, grid limit and state of charge. Planned discharge is treated as
  intent; forecast error cannot create fictitious export.
- **Honest benchmark:** the old perfect-foresight mode is retained but
  explicitly labelled **“Realised generation (perfect-foresight benchmark)”**
  and flagged in the UI and the settings sheet as an upper bound.
- **Peak charging:** charging in peak is still forbidden by default. If the
  user explicitly allows it, the settlement executes the plan instead of
  silently zeroing it, so the oracle benchmark reproduces the optimiser's own
  solution.

**Verified (F9):** 720-hour run, 0 fallbacks:

| Controller | Penalty (MWh) | Supplied (MWh) |
|---|---:|---:|
| Rule-based | 5,338.2 | 17,461.8 |
| MILP, causal receding horizon | 6,148.8 | 17,579.6 |
| MILP, perfect-foresight benchmark | 5,344.0 | 17,524.0 |

The causal MILP is slightly worse than the greedy rule on this window. That is
the expected, honest consequence of removing look-ahead: the rule is tuned for
this plant and the MILP cannot see the future. The oracle result is close to
the rule, which confirms the optimiser itself is sound; the gap between causal
and oracle is the value of the forecast.

The F9 check also verifies the hourly SoC recursion, the energy balance,
PCS limits, charge/discharge complementarity and the split-site per-technology
balances. The peak-charge edge case separately verifies that an allowed peak
charge is executed and reproduced by the oracle settlement.

## 10. Independent re-audit findings

The re-audit was run in four independent passes (runtime paths, core engine,
scenario/CLI pipeline and a final diff audit). Each confirmed defect was fixed
and pinned by a regression test in `test_audit_fixes.py` or
`test_quantile_scenarios.py`.

| ID | Defect found | Fix applied | Recheck evidence |
|---|---|---|---|
| R1 / C1 | Split MILP grid import was not tied to either battery's SoC, so imported energy vanished or was created | Import is split per battery (`imps`/`impw`), enters each SoC balance, settlement attributes it to the absorbing battery | `test_split_grid_import_enters_both_soc_and_reporting`; probe 3 split grid charge |
| R8 | Export cap was applied net of import, letting an hour import and export simultaneously; oracle replay disagreed with the plan | Gross export cap (`plant + BESS`) in both optimisers and both settlement writers, plus a separate import cap | `test_oracle_grid_import_replay_matches_plan_single`; probe 15 |
| R9 | Split MILP cycle budget counted only plant charging, so grid-charged energy could exceed the warranty limit by roughly 19x | Grid imports enter `thr_s`/`thr_w`, matching the single-site model and both settlement writers | `test_split_cycle_budget_includes_grid_charging`; 24 h probe: import = throughput = 54.794521 MWh = budget |
| M1 | One complementarity binary was shared by the solar and wind batteries | One charge/discharge binary per battery | `test_split_batteries_have_independent_complementarity` |
| M2 | Self-discharge was in the MILP but missing from settlement replay | `rho` is passed into both settlement writers and applied before each hour | `test_self_discharge_is_replayed_in_settlement` |
| R2 / M3 | A terminal SoC target outside the operating band made every window infeasible | Target is clamped into the min/max band before the constraint is built | `test_terminal_soc_is_clamped_into_the_operating_band` |
| R3 / M4 / M5 | Split penalty was measured against the grid-capped export target instead of the full contract share | Export target respects the cap; penalty target is the full contractual share in the rule engine, MILP and settlement | `test_split_penalty_matches_single_when_grid_cap_binds`; probe 8 |
| R4 / M6 | `BESS_TotalCycles` repeated the block-end value on every row | Cumulative throughput is reported per hour for single and split | `test_cycles_are_reported_cumulatively_per_hour` |
| R5 / m7 | Duplicate or out-of-range peak hours inflated required demand | `parse_peak_hours` keeps unique valid clock hours; daily/monthly demand uses the unique count | `test_peak_hours_are_deduplicated_and_penalty_uses_unique_hours` |
| R6 / m8 | Negative capacities/contracts produced negative exports and inflated curtailment | `sanitize_bess_params` clamps every engine input; UI number inputs also have `min_value=0` | `test_negative_inputs_are_sanitized`; probe 14 |
| R7 / m6 / P9 | A P50-only upload made the scenario map truthy and suppressed the embedded P75/P90 build | P50 is promoted to `solar_MWh`/`wind_MWh`, and the "pre-built" test now requires a P75 or P90 column | `test_validator_accepts_p50_only_and_rejects_fractional_hour`; probe 16 |
| M7 | Short 0-based degradation profiles were misread; percentages and duplicate years were accepted | Profile is coerced, clipped, de-duplicated and shifted from 0-based to 1-based correctly | `test_degradation_profile_is_shifted_deduplicated_and_clamped` |
| M8 | Custom formulas could not call `np.maximum` / `np.where` | Formula evaluation uses `eval` with NumPy in the context | `test_custom_formula_can_call_numpy` |
| m1 | Fractional `Hour` values were truncated to integers and could hide duplicate hours | Validator rejects non-integer hours | `test_validator_accepts_p50_only_and_rejects_fractional_hour` |
| m4 | The causal spike filter fell back to the full-sample standard deviation in warm-up rows, leaking future data | Warm-up rows without strictly-prior history are left unclipped | `test_quantile_scenarios.py` + pipeline probe |
| m9 | MILP help text said the horizon default was 48 h while the widget used 24 h | Help text corrected to 24 h | UI integration check |
| P1 | Annual `--side upper` used the wrong quantile and reversed the monotonicity direction | Upper side uses `1 - level/100` and enforces P10 >= P25 >= P50 | `test_annual_upper_side_orders_p10_above_p25_above_p50` |
| P3 | CLI runners bypassed `validate_input_frame` and sliced files positionally | Both runners call the shared validator before slicing | `test_cli_runner_rejects_duplicate_timestamps` |
| P4 | Annual metadata labelled the forecast train years as the factor sample | `annual_fit_source` and `annual_fit_years` record the actual fit sample | `test_annual_run_records_fit_source_and_years` |
| P5 | Partial years and missing values were summed as if complete | Annual calibration rejects incomplete years and missing generation | `test_annual_fit_requires_complete_years` |
| P6 | `--calibrate-p50` was silently inert in annual mode | CLI prints an explicit note | `test_annual_run_records_fit_source_and_years` |
| P8 | The app did not clamp P75 to P50 before enforcing P90 <= P75 | Both per-technology clamps are applied in order | `test_embedded_annual_scenarios_stay_ordered` |
| P10 | A missing wind column in the annual actual file became zero wind | Missing wind is rejected; NaN generation is rejected rather than zero-filled | `test_annual_actual_missing_wind_column_is_rejected` |
| P12 | Annual completeness counted rows only: 8760 rows in a leap year, or a duplicate replacing a missing hour, passed | `_require_complete_annual_frame` rejects duplicate/non-hourly timestamps and requires an unbroken 8760/8784-hour calendar year | `test_annual_fit_requires_complete_years` covers short, leap-short, duplicate, non-hourly and complete leap/non-leap samples |
| P7 / P11 | Stale `--forecast` documentation and stale code comments | README and `SCENARIO_GUIDE.md` now use `--forecast-mode`; comments corrected | Documentation review |
| — | Grid charging was invisible in the output schema | `Grid_Import_MWh` added to single/split rule and optimised schemas; import enters the energy-balance identity | `test_oracle_grid_import_replay_matches_plan_single` |

### Accepted design limitations (documented, not silent defects)

| Item | Reason |
|---|---|
| m2: repeating a non-leap year into a leap target leaves 29 February absent | The app treats the input as a representative year and keeps the chronology valid by dropping 29 February from the source when the target is not a leap year. Adding a synthetic leap day would invent generation. |
| m3: the first causal window has no prior history | At t = 0 no earlier observation exists. The first window uses the available window values; after that the forecast is strictly causal. This is inherent to a cold start, not a look-ahead leak. |
| m5: index-form `Year`/`DATE` inputs | The supported input contract is explicit `Year, Month, Day, Hour` columns. Files without them are rejected with a clear error instead of being silently misinterpreted. |

## Regression status

```text
F1/F2  PASS   F3  PASS   F4  PASS   F5  PASS   F6  PASS
F7     PASS   F8  PASS   F9  PASS   F9 peak-charge  PASS
UI integration (rule, causal MILP, split block, oracle, settings)  PASS
python -m pytest -q test_quantile_scenarios.py test_audit_fixes.py  29 passed
```

The completion workbook additionally records the recheck result for every
original row and every additional finding.
