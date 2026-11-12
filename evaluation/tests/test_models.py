import numpy as np
import pandas as pd
import pytest

from evaluation import models
from evaluation.cohort import Cohort, Split, VITALS, design_from_raw


def make_cohort(n: int = 1200, seed: int = 0) -> Cohort:
    rng = np.random.default_rng(seed)
    names = list(VITALS)
    complaints = ["a", "b", "c"]
    scaling = {
        "age": {"mean": 50.0, "sd": 20.0},
        "vitals": {v: {"mean": 0.0, "sd": 1.0} for v in names},
    }
    splits = {}
    for name, share in (("old", 0.4), ("new", 0.4), ("test", 0.2)):
        m = int(n * share)
        X = np.column_stack(
            [
                rng.normal(50, 20, m),
                rng.integers(0, 2, m),
                rng.integers(0, 2, (m, len(complaints))),
            ]
        )
        raw = rng.normal(0, 1, (m, len(names)))
        raw[rng.random(raw.shape) < 0.2] = np.nan
        V, R = design_from_raw(raw, np.zeros(len(names)), np.ones(len(names)))
        eta = -2.0 + 0.01 * X[:, 0] + 0.4 * X[:, 1] + 0.2 * np.nan_to_num(raw[:, 0])
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
        visits = pd.DataFrame(
            {
                "lopnr_swed": np.arange(m) + (0 if name == "old" else 10_000 if name == "new" else 20_000),
                "ed_hospital": rng.choice(["h1", "h2"], m),
                "complaint": rng.choice(complaints, m),
                "died30": y,
            }
        )
        splits[name] = Split(
            visits=visits,
            X_base=X,
            V=V,
            R=R,
            y=y,
            raw=raw,
        )
    return Cohort(
        splits=splits,
        base_feature_names=["age", "male"] + [f"sct_{c}" for c in complaints],
        vital_feature_names=[f"{v}_{s}" for v in names for s in ("z", "z2")],
        vital_names=names,
        complaints=complaints,
        scaling=scaling,
    )


@pytest.fixture(scope="module")
def cohort():
    return make_cohort()


def test_fit_all_shapes_and_ranges(cohort):
    fitted = models.fit_all(cohort)
    assert set(fitted) == {"M_ref", "M_base", "A", "B", "C", "M_direct"}
    for model in fitted.values():
        p = model.predict(cohort["test"])
        assert p.shape == (len(cohort["test"].y),)
        assert np.all(np.isfinite(p))
        assert p.min() >= 0 and p.max() <= 1


def test_models_score_identical_rows(cohort):
    fitted = models.fit_all(cohort)
    lengths = {model.predict(cohort["test"]).shape[0] for model in fitted.values()}
    assert lengths == {len(cohort["test"].y)}


def test_strategy_a_is_offset_plus_vitals(cohort):
    ref = models.fit_m_ref(cohort)
    a = models.fit_strategy_a(cohort, ref)
    test = cohort["test"]
    delta = a.linear_predictor(test) - ref.linear_predictor(test)
    expected = models.vital_features(test) @ a.theta
    assert np.allclose(delta, expected)


def test_strategy_c_prior_dimensions(cohort):
    ref = models.fit_m_ref(cohort)
    c = models.fit_strategy_c(cohort, ref)
    expected = ref.n_params + models.vital_features(cohort["new"]).shape[1]
    assert c.n_params == expected


def test_fits_are_deterministic(cohort):
    a = models.fit_m_ref(cohort)
    b = models.fit_m_ref(cohort)
    assert np.allclose(a.theta, b.theta)
