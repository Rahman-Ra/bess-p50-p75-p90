"""Regression tests for the independent audit fixes in ``bess_app_v6.py``.

Run with::

    pytest -q test_audit_fixes.py

These tests pin the behaviours that were wrong in the audited baseline:
grid-import accounting, oracle settlement replay, split complementarity, the
split contract penalty base, terminal-SoC clamping, self-discharge replay,
per-hour cycle reporting, the grid-charged cycle budget, negative-input
sanitisation, custom NumPy formulas, P50-only uploads, peak-hour
de-duplication, fractional-hour validation and the degradation-profile
cleanup.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_PATH = Path(__file__).with_name("bess_app_v6.py")


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("bess_app_v6_audit", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Keep the headless tests free of Streamlit progress-bar noise.
    mod._progressive_bar = lambda *a, **k: None
    mod._update_bar = lambda *a, **k: None
    mod._close_bar = lambda *a, **k: None
    return mod


def _hours(periods, solar=0.0, wind=0.0, start="2024-01-01 00:00"):
    ts = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame(
        {
            "Year": ts.year,
            "Month": ts.month,
            "Day": ts.day,
            "Hour": ts.hour,
            "solar_MWh": np.full(periods, float(solar)),
            "wind_MWh": np.full(periods, float(wind)),
        }
    )


def _single(app, **kw):
    vals = dict(
        solar_cap_mw=10.0,
        wind_cap_mw=0.0,
        contract_mw=5.0,
        bess_energy_mwh=20.0,
        pcs_power_mw=10.0,
        eta_charge=0.97,
        eta_discharge=0.96,
        min_soc_pct=0.0,
        max_soc_pct=90.0,
        initial_soc_pct=0.0,
        peak_hours=[19, 20],
        grid_limit_mw=None,
        mode="single",
    )
    vals.update(kw)
    return app.BESSParams(**vals)


def _split(app, **kw):
    vals = dict(
        solar_cap_mw=100.0,
        wind_cap_mw=100.0,
        contract_mw=100.0,
        bess_energy_mwh=0.0,
        pcs_power_mw=0.0,
        solar_bess_energy_mwh=1000.0,
        solar_pcs_power_mw=50.0,
        wind_bess_energy_mwh=1000.0,
        wind_pcs_power_mw=50.0,
        eta_charge=0.97,
        eta_discharge=0.96,
        min_soc_pct=0.0,
        max_soc_pct=90.0,
        initial_soc_pct=50.0,
        peak_hours=[19, 20],
        grid_limit_mw=None,
        mode="split",
        solar_contract_share=0.5,
    )
    vals.update(kw)
    return app.BESSParams(**vals)


def _opt(app, **kw):
    vals = dict(
        horizon_hours=24,
        step_hours=24,
        forecast_mode="oracle",
        forecast_method="seasonal_naive",
        solver="auto",
        time_limit_s=20.0,
        mip_gap=0.0,
        msg=False,
        w_penalty=1000.0,
        w_curtail=1.0,
        w_degradation=0.0,
        grid_import_price=0.0,
        tariff=None,
        allow_grid_charging=False,
        no_charge_in_peak=True,
        self_discharge_per_hour=0.0,
        terminal_soc_pct=None,
        w_terminal_soc=0.0,
        max_equivalent_cycles_per_year=None,
        force_complementarity=None,
        degeneracy_epsilon=1e-6,
        spike_filter_window=0,
        spike_filter_k=2.5,
    )
    vals.update(kw)
    return app.OptimizerParams(**vals)


def _prepared(app, frame, peak_hours):
    frame = app.ensure_hour_int(frame.copy().reset_index(drop=True))
    frame["solar_MWh"] = frame["solar_MWh"].astype(float)
    frame["wind_MWh"] = frame["wind_MWh"].astype(float)
    frame["gen_MWh"] = frame["solar_MWh"] + frame["wind_MWh"]
    frame["is_peak"] = frame["Hour"].isin(peak_hours).astype(int)
    return frame


def _single_result_frame(app, frame, capacity):
    out = frame.copy()
    for col, default in (
        ("BESS_Capacity_MWh", capacity),
        ("BESS_SOC_MWh", 0.0),
        ("BESS_SOC_pct", 0.0),
        ("BESS_Charge_in_MWh", 0.0),
        ("BESS_Charge_to_SOC_MWh", 0.0),
        ("BESS_Charge_Loss_MWh", 0.0),
        ("BESS_Discharge_MWh", 0.0),
        ("BESS_TotalCycles", 0.0),
        ("Grid_Import_MWh", 0.0),
        ("solar_to_grid_MWh", 0.0),
        ("wind_to_grid_MWh", 0.0),
        ("GenPlant_to_Grid_MWh", 0.0),
        ("BESS_to_Grid_MWh", 0.0),
        ("Total_Supplied_MWh", 0.0),
        ("Peak_Time_Energy_Supplied_MWh", 0.0),
        ("Offpeak_Supplied_MWh", 0.0),
        ("Curtailed_MWh", 0.0),
        ("Penalty_MWh", 0.0),
    ):
        out[col] = float(default)
    return out


def _split_result_frame(app, frame):
    out = frame.copy()
    for col in (
        "solar_bess_soc_mwh",
        "wind_bess_soc_mwh",
        "solar_bess_charge_mwh",
        "wind_bess_charge_mwh",
        "solar_bess_discharge_mwh",
        "wind_bess_discharge_mwh",
        "solar_bess_total_cycles",
        "wind_bess_total_cycles",
        "Grid_Import_MWh",
        "solar_to_grid_MWh",
        "wind_to_grid_MWh",
        "GenPlant_to_Grid_MWh",
        "BESS_to_Grid_MWh",
        "Total_Supplied_MWh",
        "Curtailed_MWh",
        "Penalty_MWh",
    ):
        out[col] = 0.0
    return out


def test_validator_accepts_p50_only_and_rejects_fractional_hour(app):
    frame = _hours(24)
    frame["solar_p50_MWh"] = frame.pop("solar_MWh")
    frame["wind_p50_MWh"] = frame.pop("wind_MWh")
    clean, errors, warnings = app.validate_input_frame(frame)
    assert not errors, errors
    assert {"solar_MWh", "wind_MWh"} <= set(clean.columns)
    assert any("P50 generation basis" in w for w in warnings)
    # A P50-only upload must still be treated as "not pre-built scenarios".
    assert app.scenario_columns_present(clean) is False

    bad = _hours(3)
    bad["Hour"] = bad["Hour"].astype(float)
    bad.loc[1, "Hour"] = 5.5
    _, errors, _ = app.validate_input_frame(bad)
    assert any("non-integer Hour" in e for e in errors)


def test_peak_hours_are_deduplicated_and_penalty_uses_unique_hours(app):
    assert app.parse_peak_hours("19,19,20,25,-1") == [19, 20]
    frame = _hours(24)
    frame["is_peak"] = frame["Hour"].isin([19, 20]).astype(int)
    frame["Total_Supplied_MWh"] = 0.0
    p = _single(app, contract_mw=5.0, peak_hours=[19, 20])
    daily = app.make_daily_penalty(frame, p)
    # Two unique peak hours -> 10 MWh required, not 20 from a duplicated list.
    assert daily["Required_demand_MWh"].iloc[0] == pytest.approx(10.0)
    assert daily["Penalty_MWh"].iloc[0] == pytest.approx(10.0)


def test_negative_inputs_are_sanitized(app):
    frame = _hours(24, solar=5.0)
    p = _single(
        app,
        solar_cap_mw=-10.0,
        wind_cap_mw=-10.0,
        contract_mw=-50.0,
        bess_energy_mwh=-20.0,
        pcs_power_mw=-10.0,
        peak_hours=[19, 20],
    )
    clean = app.sanitize_bess_params(p)
    assert clean.solar_cap_mw == 0.0
    assert clean.wind_cap_mw == 0.0
    assert clean.contract_mw == 0.0
    assert clean.bess_energy_mwh == 0.0
    assert clean.pcs_power_mw == 0.0
    assert clean.peak_hours == [19, 20]

    out, _ = app.simulate_single_site(frame, p, 1.0)
    assert (out["Total_Supplied_MWh"] >= 0.0).all()
    assert (out["GenPlant_to_Grid_MWh"] >= 0.0).all()
    assert (out["Curtailed_MWh"] >= -1e-9).all()
    assert out["Curtailed_MWh"].sum() <= out["gen_MWh"].sum() + 1e-6


def test_terminal_soc_is_clamped_into_the_operating_band(app):
    frame = _prepared(app, _hours(24), [19, 20])
    p = _single(app, initial_soc_pct=50.0, grid_limit_mw=None)
    o = _opt(
        app,
        allow_grid_charging=True,
        grid_import_price=10.0,
        terminal_soc_pct=150.0,
    )
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_bin, _ = app.needs_complementarity(p, o)
    sol = app._optimize_single_window(frame, p, o, 1.0, 9.0, solver, use_bin)
    assert sol is not None
    # Clamped to max_soc_pct (90 %) instead of making every window infeasible.
    assert sol["soc"][-1] <= 18.0 + 1e-6
    assert sol["soc"][-1] >= 18.0 - 1e-4


def test_self_discharge_is_replayed_in_settlement(app):
    frame = _prepared(app, _hours(24), [19, 20])
    p = _single(app, bess_energy_mwh=100.0, pcs_power_mw=10.0, initial_soc_pct=50.0)
    o = _opt(app, self_discharge_per_hour=0.01)
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_bin, _ = app.needs_complementarity(p, o)
    sol = app._optimize_single_window(frame, p, o, 1.0, 45.0, solver, use_bin)
    assert sol is not None
    out = _single_result_frame(app, frame, 100.0)
    soc, _ = app._write_single_block(
        out, 0, len(frame), sol, p, 1.0, 45.0, 0.0, oracle=True, rho=0.99
    )
    assert soc == pytest.approx(float(sol["soc"][-1]), abs=1e-6)


def test_oracle_grid_import_replay_matches_plan_single(app):
    frame = _prepared(app, _hours(24, solar=10.0), [19, 20])
    p = _single(app, contract_mw=5.0, grid_limit_mw=5.0, initial_soc_pct=0.0)
    o = _opt(
        app,
        allow_grid_charging=True,
        no_charge_in_peak=False,
        grid_import_price=-10.0,
    )
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_bin, reason = app.needs_complementarity(p, o)
    assert use_bin and "negative" in reason
    sol = app._optimize_single_window(frame, p, o, 1.0, 0.0, solver, use_bin)
    assert sol is not None
    out = _single_result_frame(app, frame, 20.0)
    soc, _ = app._write_single_block(
        out, 0, len(frame), sol, p, 1.0, 0.0, 0.0, oracle=True
    )
    assert out["BESS_Charge_in_MWh"].sum() == pytest.approx(
        float(sol["charge_in"].sum()), abs=1e-4
    )
    assert out["Grid_Import_MWh"].sum() == pytest.approx(
        float(sol["import"].sum()), abs=1e-4
    )
    assert out["BESS_Discharge_MWh"].sum() == pytest.approx(
        float(sol["discharge"].sum()), abs=1e-4
    )
    assert soc == pytest.approx(float(sol["soc"][-1]), abs=1e-4)
    # Gross export respects the connection limit even while importing.
    gross = (
        out["GenPlant_to_Grid_MWh"].to_numpy()
        + out["BESS_to_Grid_MWh"].to_numpy()
    )
    assert np.all(gross <= 5.0 + 1e-4)


def test_split_grid_import_enters_both_soc_and_reporting(app):
    frame = _prepared(app, _hours(3), [19, 20])
    p = _split(
        app,
        solar_bess_energy_mwh=10.0,
        wind_bess_energy_mwh=10.0,
        solar_pcs_power_mw=5.0,
        wind_pcs_power_mw=5.0,
        initial_soc_pct=0.0,
        grid_limit_mw=10.0,
        peak_hours=[],
    )
    o = _opt(
        app,
        allow_grid_charging=True,
        no_charge_in_peak=False,
        grid_import_price=-10.0,
    )
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_bin, _ = app.needs_complementarity(p, o)
    sol = app._optimize_split_window(
        frame, p, o, 10.0, 10.0, 0.0, 0.0, solver, use_bin
    )
    assert sol is not None
    assert sol["imp"].sum() > 0.0
    assert sol["es"][-1] > 0.0 and sol["ew"][-1] > 0.0
    out = _split_result_frame(app, frame)
    ss, ws, _, _ = app._write_split_block(
        out, 0, len(frame), sol, p, 10.0, 10.0, 0.0, 0.0, oracle=True
    )
    assert out["Grid_Import_MWh"].sum() == pytest.approx(
        float(sol["imp"].sum()), abs=1e-4
    )
    assert ss == pytest.approx(float(sol["es"][-1]), abs=1e-4)
    assert ws == pytest.approx(float(sol["ew"][-1]), abs=1e-4)


def test_split_cycle_budget_includes_grid_charging(app):
    # Zero generation, negative import price: without counting imports the
    # optimiser can import and cycle far beyond the warranty budget.
    frame = _prepared(app, _hours(24), [])
    p = _split(
        app,
        solar_bess_energy_mwh=1000.0,
        wind_bess_energy_mwh=1000.0,
        solar_pcs_power_mw=50.0,
        wind_pcs_power_mw=50.0,
        initial_soc_pct=0.0,
        peak_hours=[],
    )
    o = _opt(
        app,
        allow_grid_charging=True,
        no_charge_in_peak=False,
        grid_import_price=-10.0,
        max_equivalent_cycles_per_year=10.0,
    )
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    use_bin, _ = app.needs_complementarity(p, o)
    sol = app._optimize_split_window(
        frame, p, o, 1000.0, 1000.0, 0.0, 0.0, solver, use_bin
    )
    assert sol is not None
    assert sol["imps"].sum() > 0.0 and sol["impw"].sum() > 0.0
    budget = 2.0 * 1000.0 * 10.0 * (24.0 / 8760.0)
    # cs/cw are plant charging, so with zero generation the throughput is
    # grid import plus discharged energy (converted to charge-side MWh).
    thr_s = sol["cs"] + sol["imps"] + sol["ds"] / p.eta_discharge
    thr_w = sol["cw"] + sol["impw"] + sol["dw"] / p.eta_discharge
    assert thr_s.sum() <= budget + 1e-4
    assert thr_w.sum() <= budget + 1e-4


def test_split_batteries_have_independent_complementarity(app):
    frame = _prepared(app, _hours(1, solar=100.0, wind=0.0), [0])
    frame.loc[0, "Hour"] = 19
    frame["is_peak"] = 1
    p = _split(app)
    o = _opt(
        app,
        horizon_hours=1,
        step_hours=1,
        force_complementarity=True,
        no_charge_in_peak=False,
    )
    solver, _ = app.resolve_solver(o.solver, o.time_limit_s, o.mip_gap, o.msg)
    sol = app._optimize_split_window(
        frame, p, o, 1000.0, 1000.0, 500.0, 500.0, solver, True
    )
    assert sol is not None
    # Solar charges while wind discharges: impossible with one shared binary.
    assert sol["cs"][0] > 1.0
    assert sol["dw"][0] > 1.0
    assert sol["cur"][0] <= 1e-4


def test_cycles_are_reported_cumulatively_per_hour(app):
    frame = _prepared(app, _hours(2, solar=10.0), [19, 20])
    p = _single(app)
    sol = {
        "charge_in": np.array([5.0, 5.0]),
        "import": np.zeros(2),
        "discharge": np.zeros(2),
        "gen_to_grid": np.zeros(2),
        "curtailed": np.zeros(2),
        "penalty": np.zeros(2),
        "soc_end": np.zeros(2),
    }
    out = _single_result_frame(app, frame, 20.0)
    app._write_single_block(out, 0, 2, sol, p, 1.0, 0.0, 0.0, oracle=True)
    assert out["BESS_TotalCycles"].iloc[1] > out["BESS_TotalCycles"].iloc[0]
    assert out["BESS_TotalCycles"].iloc[0] == pytest.approx(5.0 / (2.0 * 20.0))


def test_custom_formula_can_call_numpy(app):
    frame = _hours(3, solar=2.0)
    frame.loc[1, "solar_MWh"] = 0.0
    p = _single(app)
    formulas = pd.DataFrame(
        {
            "New column": ["clipped"],
            "Formula": ["np.where(solar_MWh > 0, np.maximum(solar_MWh, 1.0), 0.0)"],
        }
    )
    out = app.apply_custom_formulas(frame, formulas, p)
    assert "clipped" in out.columns
    assert out["clipped"].tolist() == [2.0, 0.0, 2.0]


def test_degradation_profile_is_shifted_deduplicated_and_clamped(app):
    class Upload(io.BytesIO):
        name = "degradation.csv"

    csv = "Year_index,Cap_factor\n" + "\n".join(
        f"{i},{0.99 if i == 3 else 0.95}" for i in range(25)
    )
    prof = app.load_degradation_profile(Upload(csv.encode()))
    assert list(prof.index) == list(range(1, 26))
    assert prof.iloc[0] == pytest.approx(0.95)
    # The 0..24 input is shifted to 1..25, so original year 3 becomes index 4.
    assert prof.loc[3] == pytest.approx(0.95)
    assert prof.loc[4] == pytest.approx(0.99)
    assert prof.between(0.0, 1.0).all()

    duplicate = "Year_index,Cap_factor\n1,0.90\n1,0.80\n2,95\n"
    prof2 = app.load_degradation_profile(Upload(duplicate.encode()))
    # Last duplicate wins and 95 (a percentage) is clipped to 1.0.
    assert prof2.loc[1] == pytest.approx(0.80)
    assert prof2.loc[2] == pytest.approx(1.0)

    short = "Year_index,Cap_factor\n" + "\n".join(
        f"{i},{1.0 - i / 100.0}" for i in range(5)
    )
    prof3 = app.load_degradation_profile(Upload(short.encode()))
    # A short 0..4 profile is still shifted to 1..5, then filled forward.
    assert prof3.loc[1] == pytest.approx(1.0)
    assert prof3.loc[5] == pytest.approx(0.96)
    assert prof3.loc[6] == pytest.approx(0.96)


def test_split_penalty_matches_single_when_grid_cap_binds(app):
    frame = _hours(1, solar=100.0)
    frame.loc[0, "Hour"] = 19
    frame["is_peak"] = 1
    single_p = _single(
        app,
        solar_cap_mw=100.0,
        wind_cap_mw=0.0,
        contract_mw=100.0,
        bess_energy_mwh=0.0,
        pcs_power_mw=0.0,
        grid_limit_mw=50.0,
        peak_hours=[19],
    )
    out_single, _ = app.simulate_single_site(frame, single_p, 1.0)
    split_p = _split(
        app,
        solar_bess_energy_mwh=0.0,
        solar_pcs_power_mw=0.0,
        wind_bess_energy_mwh=0.0,
        wind_pcs_power_mw=0.0,
        contract_mw=100.0,
        grid_limit_mw=50.0,
        solar_contract_share=1.0,
        peak_hours=[19],
    )
    out_split, _, _ = app.simulate_split_site(frame, split_p, 1.0, 1.0)
    assert out_single["Penalty_MWh"].iloc[0] == pytest.approx(50.0, abs=1e-6)
    assert out_split["Penalty_MWh"].iloc[0] == pytest.approx(
        out_single["Penalty_MWh"].iloc[0], abs=1e-6
    )


def test_embedded_annual_scenarios_stay_ordered(app):
    frame = _hours(24, solar=10.0, wind=5.0)
    out = app.build_generation_scenarios(frame, mode="annual")
    for tech in ("solar", "wind", "total"):
        assert (out[f"{tech}_p90_MWh"] <= out[f"{tech}_p75_MWh"] + 1e-9).all()
        assert (out[f"{tech}_p75_MWh"] <= out[f"{tech}_p50_MWh"] + 1e-9).all()


def test_cli_runner_rejects_duplicate_timestamps(app, tmp_path):
    import run_experiment

    frame = _hours(3)
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        run_experiment.load_validated_input(app, path)
