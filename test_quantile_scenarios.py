"""Tests for quantile_scenarios.py.

Run with::

    pytest -q test_quantile_scenarios.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantile_scenarios as qs


def _normal_sample(n=40_000, seed=7, bias=0.0, sd=10.0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2020-01-01", periods=n, freq="h")
    p50 = rng.uniform(0.0, 80.0, n)
    actual = np.maximum(0.0, p50 + bias + rng.normal(0.0, sd, n))
    return time, actual, p50


def test_additive_quantiles_hit_target_coverage():
    time, actual, p50 = _normal_sample()
    split = 20_000
    cal = qs.fit_exceedance_calibration(
        actual[:split], p50[:split], levels=(50, 75, 90),
        group_cols=(), method="additive", calibrate_p50=True,
    )
    scen = cal.scenario(p50[split:])
    for level, expected in ((50, 0.50), (75, 0.75), (90, 0.90)):
        coverage = float(np.mean(actual[split:] >= scen[level]))
        assert abs(coverage - expected) < 0.02, (level, coverage)


def test_fixed_haircut_is_not_a_probability_statement():
    # A fixed 0.8 x P50 is not P90; this guards against "simplifying" the
    # calibration back to a constant factor.
    time, actual, p50 = _normal_sample(seed=11, sd=10.0)
    cal = qs.fit_exceedance_calibration(
        actual[:20_000], p50[:20_000], levels=(90,), calibrate_p50=True,
    )
    calibrated = cal.scenario(p50[20_000:])[90]
    fixed = 0.8 * p50[20_000:]
    fixed_coverage = float(np.mean(actual[20_000:] >= fixed))
    calibrated_coverage = float(np.mean(actual[20_000:] >= calibrated))
    assert fixed_coverage < calibrated_coverage - 0.10


def test_scenarios_are_monotone_and_non_negative():
    time, actual, p50 = _normal_sample(n=5_000, seed=3)
    cal = qs.fit_exceedance_calibration(
        actual, p50, levels=(50, 75, 90), group_cols=("month", "hour"),
        calibrate_p50=True,
        month=time.month.to_numpy(), hour=time.hour.to_numpy(),
    )
    scen = cal.scenario(
        p50, month=time.month.to_numpy(), hour=time.hour.to_numpy()
    )
    assert np.all(scen[90] >= 0.0)
    assert np.all(scen[90] <= scen[75] + 1e-9)
    assert np.all(scen[75] <= scen[50] + 1e-9)


def test_group_calibration_uses_month_and_hour():
    time = pd.date_range("2021-01-01", periods=24 * 60, freq="h")
    month = time.month.to_numpy()
    hour = time.hour.to_numpy()
    p50 = np.full(len(time), 10.0)
    # Evening hours are much more uncertain than midday hours.
    actual = np.where(hour >= 18, 2.0, 12.0)
    cal = qs.fit_exceedance_calibration(
        actual, p50, levels=(50, 75, 90), group_cols=("hour",),
        min_samples=2, shrinkage=0.0, calibrate_p50=True, hour=hour,
    )
    midday = cal.scenario(np.array([10.0]), month=np.array([1]), hour=np.array([12]))
    evening = cal.scenario(np.array([10.0]), month=np.array([1]), hour=np.array([20]))
    assert midday[90][0] > evening[90][0]


def test_annual_factors_are_derived_from_annual_energy():
    years = np.repeat([2018, 2019, 2020, 2021, 2022], 24)
    actual = np.concatenate([
        np.full(24, 100.0), np.full(24, 90.0), np.full(24, 80.0),
        np.full(24, 110.0), np.full(24, 100.0),
    ])
    cal = qs.fit_annual_factors(actual, years, levels=(50, 75, 90))
    assert cal.factors[50] == pytest.approx(1.0)
    assert cal.factors[90] < cal.factors[75] < 1.0
    scen = cal.scenario(np.array([10.0, 20.0]))
    assert np.all(scen[90] <= scen[75])
    assert np.all(scen[75] <= scen[50])


def test_joint_scenarios_preserve_total_and_do_not_add_p90s():
    time = pd.date_range("2022-01-01", periods=500, freq="h")
    rng = np.random.default_rng(5)
    solar_p50 = rng.uniform(0.0, 50.0, len(time))
    wind_p50 = rng.uniform(0.0, 50.0, len(time))
    solar_actual = np.maximum(0.0, solar_p50 + rng.normal(0, 8, len(time)))
    wind_actual = np.maximum(0.0, wind_p50 + rng.normal(0, 8, len(time)))
    total_p50 = solar_p50 + wind_p50
    total_actual = solar_actual + wind_actual

    joint = qs.fit_exceedance_calibration(
        total_actual, total_p50, levels=(50, 75, 90), calibrate_p50=True,
    )
    out = qs.apply_joint_scenarios(solar_p50, wind_p50, joint, levels=(50, 75, 90))
    for level in (50, 75, 90):
        s, w, total = out[level]
        assert np.allclose(s + w, total)
        assert np.all(total <= out[50][2] + 1e-9)
    # The naive sum of independent P90s is not the joint P90 in general.
    solar_cal = qs.fit_exceedance_calibration(
        solar_actual, solar_p50, levels=(90,), calibrate_p50=True,
    )
    wind_cal = qs.fit_exceedance_calibration(
        wind_actual, wind_p50, levels=(90,), calibrate_p50=True,
    )
    naive = solar_cal.scenario(solar_p50)[90] + wind_cal.scenario(wind_p50)[90]
    assert not np.allclose(naive, out[90][2])


def test_write_battery_inputs_replaces_only_scenario_window(tmp_path):
    times = pd.date_range("2023-12-31 22:00", periods=6, freq="h")
    base = pd.DataFrame({
        "Year": times.year, "Month": times.month, "Day": times.day,
        "Hour": times.hour, "solar_MWh": [1.0] * 6, "wind_MWh": [2.0] * 6,
    })
    base_path = tmp_path / "base.csv"
    base.to_csv(base_path, index=False)

    scen = pd.DataFrame({
        "time": times[1:5],
        "solar_p50_MWh": [10.0] * 4, "solar_p75_MWh": [9.0] * 4,
        "solar_p90_MWh": [8.0] * 4,
        "wind_p50_MWh": [20.0] * 4, "wind_p75_MWh": [18.0] * 4,
        "wind_p90_MWh": [16.0] * 4,
    })
    written = qs.write_battery_inputs(
        base_path, scen, tmp_path, levels=(50, 75, 90), apply_year=None,
    )
    assert len(written) == 3
    p90 = pd.read_csv(written[2])
    # Four forecast hours are replaced; the two outside the window keep 1/2.
    assert p90["solar_MWh"].tolist() == [1.0, 8.0, 8.0, 8.0, 8.0, 1.0]
    assert p90["wind_MWh"].tolist() == [2.0, 16.0, 16.0, 16.0, 16.0, 2.0]


def test_write_battery_inputs_drops_stale_scenario_columns(tmp_path):
    times = pd.date_range("2024-01-01", periods=4, freq="h")
    base = pd.DataFrame({
        "Year": times.year, "Month": times.month, "Day": times.day,
        "Hour": times.hour,
        "solar_MWh": [999.0] * 4, "wind_MWh": [999.0] * 4,
        "solar_p75_MWh": [999.0] * 4, "wind_p75_MWh": [999.0] * 4,
    })
    base_path = tmp_path / "base_with_scenarios.csv"
    base.to_csv(base_path, index=False)

    scen = pd.DataFrame({
        "time": times,
        "solar_p50_MWh": [10.0] * 4, "solar_p75_MWh": [9.0] * 4,
        "solar_p90_MWh": [8.0] * 4,
        "wind_p50_MWh": [20.0] * 4, "wind_p75_MWh": [18.0] * 4,
        "wind_p90_MWh": [16.0] * 4,
    })
    written = qs.write_battery_inputs(
        base_path, scen, tmp_path, levels=(50, 75, 90), apply_year=None,
    )
    p75 = pd.read_csv(written[1])
    assert p75["solar_MWh"].tolist() == [9.0] * 4
    assert p75["wind_MWh"].tolist() == [18.0] * 4


def test_joint_summary_scores_written_columns(tmp_path):
    time = pd.date_range("2022-01-01", periods=600, freq="h")
    rng = np.random.default_rng(13)
    solar_p50 = rng.uniform(0.0, 50.0, len(time))
    wind_p50 = rng.uniform(0.0, 50.0, len(time))
    solar = pd.DataFrame({
        "time": time,
        "actual": np.maximum(0.0, solar_p50 + rng.normal(0, 7, len(time))),
        "pred": solar_p50,
    })
    wind = pd.DataFrame({
        "time": time,
        "actual": np.maximum(0.0, wind_p50 + rng.normal(0, 7, len(time))),
        "pred": wind_p50,
    })
    solar_path = tmp_path / "solar.csv"
    wind_path = tmp_path / "wind.csv"
    solar.to_csv(solar_path, index=False)
    wind.to_csv(wind_path, index=False)
    out = tmp_path / "out"
    rc = qs.main([
        "--solar-pred", str(solar_path), "--wind-pred", str(wind_path),
        "--train-years", "2022", "--test-years", "2022",
        "--group", "hour", "--min-samples", "5",
        "--aggregation", "joint", "--outdir", str(out),
    ])
    assert rc == 0
    hourly = pd.read_csv(out / "scenario_hourly.csv")
    summary = pd.read_csv(out / "scenario_summary.csv")
    for target, actual_col in (
        ("solar", "solar_actual_MWh"), ("wind", "wind_actual_MWh")
    ):
        row = summary[
            (summary["target"] == f"{target}_MWh")
            & (summary["level"] == "P90")
        ].iloc[0]
        expected = float(
            (hourly[actual_col] >= hourly[f"{target}_p90_MWh"]).mean()
        )
        assert row["coverage_actual_ge_scenario"] == pytest.approx(expected)


def test_cli_writes_expected_artifacts(tmp_path):
    time, actual, p50 = _normal_sample(n=4_000, seed=21)
    solar = pd.DataFrame({"time": time, "actual": actual, "pred": p50})
    solar_path = tmp_path / "solar.csv"
    solar.to_csv(solar_path, index=False)
    out = tmp_path / "out"
    rc = qs.main([
        "--solar-pred", str(solar_path),
        "--train-years", "2020", "--test-years", "2020",
        "--group", "month,hour", "--levels", "50,75,90",
        "--outdir", str(out),
    ])
    assert rc == 0
    for name in ("scenario_hourly.csv", "scenario_summary.csv",
                 "scenario_summary.json", "scenario_calibration.json"):
        assert (out / name).exists()
    payload = json.loads((out / "scenario_calibration.json").read_text())
    assert payload["mode"] == "hourly"
    assert payload["solar"]["levels"] == [50, 75, 90]


def test_annual_upper_side_orders_p10_above_p25_above_p50():
    # Annual energy deliberately spread so the quartiles cannot be confused
    # with the lower-side labels.
    years = np.repeat([2018, 2019, 2020, 2021, 2022], 24)
    actual = np.concatenate([
        np.full(24, 100.0), np.full(24, 90.0), np.full(24, 80.0),
        np.full(24, 110.0), np.full(24, 100.0),
    ])
    cal = qs.fit_annual_factors(
        actual, years, levels=(50, 25, 10), side="upper"
    )
    assert cal.side == "upper"
    assert cal.factors[50] == pytest.approx(1.0)
    assert cal.factors[10] >= cal.factors[25] >= cal.factors[50]
    scen = cal.scenario(np.array([10.0, 20.0]), levels=(50, 25, 10))
    assert np.all(scen[10] >= scen[25] - 1e-9)
    assert np.all(scen[25] >= scen[50] - 1e-9)


def test_annual_fit_requires_complete_years():
    short = pd.date_range("2021-01-01", periods=100, freq="h")
    with pytest.raises(ValueError, match="complete years"):
        qs._require_complete_annual_frame(
            pd.DataFrame({"time": short, "actual": 1.0}), label="short sample"
        )

    # 8760 rows is not a complete leap year: 31 December is missing.
    leap_short = pd.date_range("2020-01-01", periods=8760, freq="h")
    with pytest.raises(ValueError, match="complete years"):
        qs._require_complete_annual_frame(
            pd.DataFrame({"time": leap_short, "actual": 1.0}), label="leap sample"
        )

    # A duplicate can replace a missing hour without changing the row count.
    duplicate = pd.date_range("2021-01-01", periods=8760, freq="h").to_list()
    duplicate[5] = duplicate[4]
    with pytest.raises(ValueError, match="duplicate timestamp"):
        qs._require_complete_annual_frame(
            pd.DataFrame({"time": duplicate, "actual": 1.0}), label="duplicate sample"
        )

    # Non-hourly timestamps are rejected even when the row count is right.
    non_hourly = pd.date_range("2021-01-01", periods=8760, freq="h").to_list()
    non_hourly[10] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="complete years"):
        qs._require_complete_annual_frame(
            pd.DataFrame({"time": non_hourly, "actual": 1.0}), label="non-hourly sample"
        )

    # Complete non-leap and leap years are accepted.
    for start, periods in (("2021-01-01", 8760), ("2020-01-01", 8784)):
        complete = pd.date_range(start, periods=periods, freq="h")
        qs._require_complete_annual_frame(
            pd.DataFrame({"time": complete, "actual": 1.0}), label="complete sample"
        )


def test_annual_actual_missing_wind_column_is_rejected(tmp_path):
    # Two complete non-leap years so only the missing wind column is at fault.
    time = pd.date_range("2021-01-01", periods=8760 * 2, freq="h")
    solar_pred = pd.DataFrame({"time": time, "actual": 1.0, "pred": 1.0})
    wind_pred = pd.DataFrame({"time": time, "actual": 1.0, "pred": 1.0})
    annual = pd.DataFrame({"time": time, "solar_MWh": 1.0})

    solar_path = tmp_path / "solar.csv"
    wind_path = tmp_path / "wind.csv"
    annual_path = tmp_path / "annual.csv"
    solar_pred.to_csv(solar_path, index=False)
    wind_pred.to_csv(wind_path, index=False)
    annual.to_csv(annual_path, index=False)

    with pytest.raises(ValueError, match="annual_wind|wind"):
        qs.main([
            "--solar-pred", str(solar_path),
            "--wind-pred", str(wind_path),
            "--mode", "annual",
            "--annual-actual", str(annual_path),
            "--annual-solar-col", "solar_MWh",
            "--annual-min-years", "2",
            "--outdir", str(tmp_path / "out"),
        ])


def test_annual_run_records_fit_source_and_years(tmp_path, capsys):
    time = pd.date_range("2021-01-01", periods=8760 * 2, freq="h")
    # Forecast series: only the P50 shape matters for the derating.
    solar_pred = pd.DataFrame({"time": time, "actual": 1.0, "pred": 1.0})
    wind_pred = pd.DataFrame({"time": time, "actual": 1.0, "pred": 1.0})
    # Different annual mixes so the per-technology factors are not trivially 1.
    annual = pd.DataFrame({
        "time": time,
        "solar_MWh": np.where(time.year == 2021, 10.0, 8.0),
        "wind_MWh": np.where(time.year == 2021, 4.0, 5.0),
    })
    solar_path = tmp_path / "solar.csv"
    wind_path = tmp_path / "wind.csv"
    annual_path = tmp_path / "annual.csv"
    solar_pred.to_csv(solar_path, index=False)
    wind_pred.to_csv(wind_path, index=False)
    annual.to_csv(annual_path, index=False)
    out = tmp_path / "out"

    rc = qs.main([
        "--solar-pred", str(solar_path),
        "--wind-pred", str(wind_path),
        "--mode", "annual",
        "--annual-actual", str(annual_path),
        "--annual-solar-col", "solar_MWh",
        "--annual-wind-col", "wind_MWh",
        "--annual-min-years", "2",
        "--calibrate-p50",
        "--outdir", str(out),
    ])
    assert rc == 0
    payload = json.loads((out / "scenario_calibration.json").read_text())
    assert payload["annual_fit_source"] == str(annual_path)
    assert payload["annual_fit_years"] == [2021, 2022]
    # P6 - the flag is not silently ignored in annual mode.
    assert "no effect in annual mode" in capsys.readouterr().out
