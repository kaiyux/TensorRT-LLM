"""Acceptance gates use measured evidence consistently in both execution modes."""

from __future__ import annotations

import pytest

from agent_flow.workflows.perf_optimize.measurements import validate_acceptance


def _verdict(value=108.4, gain=8.4, **extra):
    return {"measured_value": value, "measured_gain_pct": gain, **extra}


def _curve(values):
    return [
        {"concurrency": point, "value": value, "tok_s_user": 10.0, "tok_s_gpu": value}
        for point, value in values
    ]


def _validate(verdict, reference=None, **options):
    return validate_acceptance(
        verdict,
        reference or {"value": 100.0},
        metric=options.pop("metric", "output_throughput"),
        required_gain=options.pop("required_gain", 5.0),
        **options,
    )


@pytest.mark.parametrize("invalid", [None, "108.4", True, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["measured_value", "reference"])
def test_acceptance_requires_positive_finite_measurements(invalid, field):
    with pytest.raises(ValueError, match="finite"):
        _validate(
            _verdict(value=invalid if field == "measured_value" else 108.4),
            {"value": invalid if field == "reference" else 100.0},
        )


@pytest.mark.parametrize("gain", [None, "8.4", True, float("nan"), float("inf")])
def test_acceptance_rejects_invalid_reported_gain(gain):
    with pytest.raises(ValueError, match="finite"):
        _validate(_verdict(gain=gain))


def test_reported_gain_cannot_conceal_a_regression():
    with pytest.raises(ValueError, match="below required"):
        _validate(_verdict(value=90.0, gain=10.0))


def test_reported_gain_must_match_actual_gain():
    with pytest.raises(ValueError, match="measured_gain_pct mismatch"):
        _validate(_verdict(value=106.0, gain=20.0))


def test_rounding_does_not_lift_gain_over_threshold():
    with pytest.raises(ValueError, match="below required"):
        _validate(_verdict(value=104.999, gain=5.0))


def test_rounding_reported_gain_does_not_fail_an_exact_threshold_measurement():
    assert _validate(
        _verdict(value=101.0, gain=0.33),
        {"value": 100.0 + 2 / 3},
        required_gain=100 / 302,
    ) == pytest.approx(100 / 302)


def test_latency_gain_improves_downward():
    assert _validate(_verdict(value=90.0, gain=10.0), metric="mean_tpot_ms") == pytest.approx(10)


def test_full_curve_coverage_is_required_even_with_focus():
    reference = {"value": 100.0, "curve": _curve([(8, 100.0), (32, 100.0)])}
    with pytest.raises(ValueError, match="does not cover"):
        _validate(
            _verdict(value=110.0, gain=10.0, curve=_curve([(8, 110.0)])),
            reference,
            points=[8, 32],
            focus=[8],
        )


def test_duplicate_curve_points_are_rejected():
    reference = {"value": 100.0, "curve": _curve([(8, 100.0), (32, 100.0)])}
    with pytest.raises(ValueError, match="unique"):
        _validate(
            _verdict(curve=_curve([(8, 110.0), (8, 110.0), (32, 110.0)])),
            reference,
            points=[8, 32],
        )


def test_nonfocus_curve_regression_still_vetoes_acceptance():
    reference = {"value": 100.0, "curve": _curve([(8, 100.0), (32, 100.0)])}
    with pytest.raises(ValueError, match="regresses concurrency 32"):
        _validate(
            _verdict(value=110.0, gain=10.0, curve=_curve([(8, 110.0), (32, 98.0)])),
            reference,
            points=[8, 32],
            focus=[8],
            allowed_regression=1.0,
        )


def test_curve_scoring_uses_mean_of_focus_gains_and_values():
    reference = {"value": 100.0, "curve": _curve([(8, 90.0), (32, 110.0)])}
    curve = _curve([(8, 126.0), (32, 112.2)])
    assert _validate(
        _verdict(value=112.2, gain=2.0, curve=curve),
        reference,
        required_gain=1.0,
        points=[8, 32],
        focus=[32],
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="below required"):
        _validate(
            _verdict(value=112.2, gain=21.0, curve=curve),
            reference,
            points=[8, 32],
            focus=[32],
        )


def test_curve_reported_value_must_match_scored_mean():
    reference = {"value": 100.0, "curve": _curve([(8, 90.0), (32, 110.0)])}
    with pytest.raises(ValueError, match="measured_value mismatch"):
        _validate(
            _verdict(value=110.0, gain=10.0, curve=_curve([(8, 99.0), (32, 121.0)])),
            reference,
            points=[8, 32],
            focus=[32],
        )


def test_curve_requires_ascending_rows_before_acceptance():
    reference = {"value": 100.0, "curve": _curve([(8, 90.0), (32, 110.0)])}
    with pytest.raises(ValueError, match="ascending"):
        _validate(
            _verdict(value=110.0, gain=10.0, curve=_curve([(32, 121.0), (8, 99.0)])),
            reference,
            points=[8, 32],
        )
