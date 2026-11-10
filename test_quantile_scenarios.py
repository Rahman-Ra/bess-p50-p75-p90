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
