import numpy as np
import pandas as pd
import pytest

from evaluation.fidelity import TABLES, fidelity_tables, privacy_table


@pytest.fixture
def builds(tmp_path):
    real, synth = tmp_path / "real", tmp_path / "synth"
    real.mkdir(), synth.mkdir()

    n = 25

    def ed(hospital):
        return pd.DataFrame({
            "ed_visit_id": np.arange(1, n + 1),
            "lopnr_swed": 10.0 + np.arange(n),
            "ed_hospital": [hospital] * n,
            "ed_arrival_datetime": pd.Timestamp("2020-01-01T10:00Z") + pd.to_timedelta(np.arange(n), unit="D"),
            "ed_age": 50.0 + np.arange(n) % 10,
            "is_male": np.arange(n) % 2,
            "ed_presented_complaint_std": np.where(np.arange(n) % 2, "A [001]", "B [002]"),
            "ed_priority_in": 1.0 + np.arange(n) % 3,
            "ed_arrival_type": np.where(np.arange(n) % 2, "walk", "ambulance"),
            "ed_exit_to": np.where(np.arange(n) % 2, "home", "ward"),
        })

    def vitals():
        return pd.DataFrame({
            "ed_visit_id": np.r_[np.arange(1, n + 1), np.arange(1, n + 1)],
            "datetime_analysis": pd.to_datetime(["2020-01-01T10:00Z"] * (2 * n)),
            "concept": ["sbp"] * n + ["pulse_rate"] * n,
            "value": np.r_[120 + np.arange(n), 80 + np.arange(n)],
            "implausible": [False] * (2 * n),
        })

    for d, h in ((real, "h1"), (synth, "h1")):
        ed(h).to_parquet(d / "ed_visits.parquet", index=False)
        vitals().to_parquet(d / "vitals.parquet", index=False)
        pd.DataFrame({"term_name": ["L1", "L2"]}).to_parquet(d / "lab_meas.parquet", index=False)
        pd.DataFrame({"icd10_3": ["A01", "A02"]}).to_parquet(d / "diagnoses.parquet", index=False)
    return real, synth


def test_fidelity_tables_summarises_distributions_and_correlations(builds):
    summary, detail = fidelity_tables(*builds)
    assert set(detail) >= {"dist_ed_hospital", "dist_complaint", "vital_observed", "vital_correlations"}
    assert {"row_count", "category", "vital_quantile", "vitals_mean_abs_correlation_diff"} <= set(summary["check"])
    assert set(summary.loc[summary["check"] == "row_count", "group"]) == set(TABLES)


def test_privacy_detects_exact_key_reuse(builds):
    table = privacy_table(*builds)
    row = table[table["check"] == "exact_key_match_share"].iloc[0]
    assert row["synth"] == 1.0
