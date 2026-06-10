"""Significance tests on synthetic data (PR-22 DoD)."""

from productrank.evaluation.significance import paired_significance


def _series(values):
    return {f"q{i}": v for i, v in enumerate(values)}


def test_clear_improvement_is_significant():
    # B beats A on every query (by varying, realistic margins) → significant lift.
    a = _series([0.10, 0.20, 0.15, 0.25, 0.30, 0.20, 0.18, 0.22])
    b = _series([0.28, 0.42, 0.30, 0.49, 0.55, 0.41, 0.33, 0.40])
    res = paired_significance(a, b)
    assert res.mean_diff > 0
    assert res.p_value < 0.05
    assert res.ci_low > 0
    assert res.significant


def test_constant_nonzero_diff_is_significant():
    # Edge case: identical positive gap on every query → zero variance, certain effect.
    a = _series([0.2, 0.3, 0.25, 0.4, 0.1])
    b = _series([0.4, 0.5, 0.45, 0.6, 0.3])
    res = paired_significance(a, b)
    assert res.significant
    assert res.p_value == 0.0


def test_no_difference_is_not_significant():
    a = _series([0.2, 0.3, 0.25, 0.4, 0.1])
    b = dict(a)  # identical
    res = paired_significance(a, b)
    assert res.mean_diff == 0
    assert not res.significant


def test_noisy_small_lift_not_significant():
    # Tiny inconsistent differences should not clear the bar.
    a = _series([0.2, 0.5, 0.1, 0.4, 0.3, 0.45, 0.05, 0.25])
    b = _series([0.22, 0.48, 0.13, 0.38, 0.31, 0.40, 0.09, 0.27])
    res = paired_significance(a, b)
    assert not res.significant


def test_only_shared_queries_compared():
    a = {"q1": 0.1, "q2": 0.2, "q3": 0.3}
    b = {"q2": 0.4, "q3": 0.5, "q4": 0.6}  # q1/q4 not shared
    res = paired_significance(a, b)
    assert res.n == 2  # q2, q3
