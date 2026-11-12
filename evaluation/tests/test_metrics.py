import numpy as np

from evaluation.metrics import (
    all_metrics,
    auc_score,
    brier_skill,
    calibration_intercept_slope,
    macro_auc,
    nagelkerke,
    oe_ratio,
    bootstrap_ci,
)


def test_perfect_and_chance_auc():
    y = np.array([0, 0, 1, 1])
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert abs(auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12


def test_calibration_recovers_intercept_and_slope():
    rng = np.random.default_rng(0)
    n = 20000
    lp = rng.normal(-1.0, 0.7, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    p = 1 / (1 + np.exp(-lp))
    alpha, beta = calibration_intercept_slope(y, p)
    assert abs(alpha) < 0.1
    assert abs(beta - 1.0) < 0.1


def test_oe_bss_nagelkerke_ranges():
    rng = np.random.default_rng(1)
    n = 5000
    lp = rng.normal(-2.0, 0.5, n)
    p = 1 / (1 + np.exp(-lp))
    y = rng.binomial(1, p)
    assert oe_ratio(y, p) > 0
    assert brier_skill(y, p) <= 1
    r2 = nagelkerke(y, p)
    assert 0 <= r2 <= 1


def test_macro_auc_skips_small_groups():
    y = np.array([0, 1] * 6 + [0, 1] + [0, 0, 1, 1])
    p = np.array([0.2, 0.8] * 6 + [0.3, 0.7] + [0.2, 0.4, 0.6, 0.8])
    groups = np.array(["a"] * 12 + ["tiny"] * 2 + ["b"] * 4)
    value, n_groups = macro_auc(y, p, groups, min_events=5)
    assert np.isfinite(value)
    assert n_groups == 1


def test_all_metrics_keys():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.2, 500)
    p = rng.uniform(0, 1, 500)
    out = all_metrics(y, p, groups=np.zeros(500))
    assert set(out) == {
        "micro_auc", "macro_auc", "oe_ratio", "cal_intercept", "cal_slope",
        "log_loss", "brier", "bss", "nagelkerke",
    }


def test_bootstrap_reproducible():
    rng = np.random.default_rng(3)
    n = 400
    patients = np.repeat(np.arange(200), 2)
    y = rng.binomial(1, 0.2, n)
    p = rng.uniform(0, 1, n)
    a = bootstrap_ci(y, p, patients, B=20, seed=7)
    b = bootstrap_ci(y, p, patients, B=20, seed=7)
    assert a["micro_auc"] == b["micro_auc"]
    assert a["log_loss"]["lo"] <= a["log_loss"]["value"] <= a["log_loss"]["hi"]
