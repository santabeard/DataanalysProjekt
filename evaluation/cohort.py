#!/usr/bin/env python3
"""Load the built SWED cohort and construct the modelling data.

Produces one row per ED visit with:

- ``died30``     outcome, death 0-30 days after arrival
- ``X_base``     age (standardised), sex, chief-complaint dummies
- ``V``          vital design ``(z, z**2)`` per selected vital
- ``R``          observation mask, one column per vital

Patients are split 40/40/20 into ``D_old`` / ``D_new`` / ``test`` with a fixed
seed. Scaling constants come from ``D_old`` for the base features and from
``D_new`` for the vitals, matching ``data_pipeline/07_compare_models.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VITALS = {
    "sbp": "systolic_bp",
    "pulse": "pulse_rate",
    "rr": "resp_rate",
    "spo2": "spo2",
    "temp": "body_temp",
}
CONCEPT_ALIASES = {"heart_rate": "pulse_rate"}
TOP_COMPLAINTS = 8
SPLIT = {"old": 0.4, "new": 0.4, "test": 0.2}
SEED = 42
AGE_MIN, AGE_MAX = 18, 109
MIN_CELL = 10


@dataclass
class Split:
    visits: pd.DataFrame
    X_base: np.ndarray
    V: np.ndarray
    R: np.ndarray
    y: np.ndarray


@dataclass
class Cohort:
    splits: dict[str, Split]
    base_feature_names: list[str]
    vital_feature_names: list[str]
    vital_names: list[str]
    complaints: list[str]
    scaling: dict[str, dict[str, float]] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Split:
        return self.splits[name]

    @property
    def all_visits(self) -> pd.DataFrame:
        return pd.concat([s.visits for s in self.splits.values()], ignore_index=True)


def round5(n: float) -> int:
    return int(5 * round(float(n) / 5))


def _load_visits(built: Path) -> pd.DataFrame:
    ed = pd.read_parquet(
        built / "ed_visits.parquet",
        columns=[
            "ed_visit_id",
            "lopnr_swed",
            "ed_hospital",
            "ed_arrival_datetime",
            "ed_age",
            "is_male",
            "ed_presented_complaint_std",
        ],
    )
    ed = ed[ed["ed_age"].between(AGE_MIN, AGE_MAX) & ed["is_male"].notna()].copy()
    ed["age"] = ed["ed_age"].astype(int)
    ed["male"] = ed["is_male"].astype(int)
    ed["complaint"] = ed["ed_presented_complaint_std"].str.extract(r"\[(\d+)\]", expand=False)

    deaths = pd.read_parquet(built / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
    death = pd.to_datetime(ed["lopnr_swed"].map(deaths.set_index("lopnr_swed")["date_death_minimp"]))
    arrival = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    days = (death - arrival).dt.days
    ed = ed[~(days < 0)].copy()
    ed["days_to_death"] = days[~(days < 0)].to_numpy()
    ed["died30"] = ed["days_to_death"].between(0, 30).astype(int)
    return ed.reset_index(drop=True)


def _attach_vitals(ed: pd.DataFrame, built: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    v = pd.read_parquet(
        built / "vitals.parquet",
        columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"],
    )
    v["concept"] = v["concept"].replace(CONCEPT_ALIASES)
    v = v[
        v["ed_visit_id"].notna()
        & v["value"].notna()
        & ~v["implausible"]
        & v["concept"].isin(VITALS.values())
    ].copy()
    v["ed_visit_id"] = v["ed_visit_id"].astype("int64")
    first = (
        v.sort_values("datetime_analysis")
        .groupby(["ed_visit_id", "concept"])["value"]
        .first()
        .unstack("concept")
    )

    raw = np.full((len(ed), len(VITALS)), np.nan)
    for j, (name, concept) in enumerate(VITALS.items()):
        if concept in first:
            raw[:, j] = ed["ed_visit_id"].map(first[concept]).to_numpy()
    R = (~np.isnan(raw)).astype(float)
    return ed, raw, R


def _split_patients(ed: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    patients = rng.permutation(ed["lopnr_swed"].unique())
    cuts = np.cumsum([int(share * len(patients)) for share in SPLIT.values()])[:-1]
    role = {p: name for name, part in zip(SPLIT, np.split(patients, cuts)) for p in part}
    return ed["lopnr_swed"].map(role)


def _base_design(
    d: pd.DataFrame,
    complaints: list[str],
    age_mean: float,
    age_sd: float,
) -> tuple[np.ndarray, list[str]]:
    names = ["age", "male"] + [f"sct_{c}" for c in complaints]
    X = np.empty((len(d), len(names)), dtype=float)
    X[:, 0] = (d["age"].to_numpy() - age_mean) / age_sd
    X[:, 1] = d["male"].to_numpy()
    complaint = d["complaint"].to_numpy()
    for j, c in enumerate(complaints, start=2):
        X[:, j] = (complaint == c).astype(float)
    return X, names


def _vital_design(
    raw: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    cols: list[np.ndarray] = []
    for j, name in enumerate(VITALS):
        z = (raw[:, j] - mean[j]) / sd[j]
        z = np.where(np.isnan(z), 0.0, z)
        cols += [z, z**2]
        names += [f"{name}_z", f"{name}_z2"]
    return np.column_stack(cols), names


def _safe_sd(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x, axis=0)
    return np.where((sd == 0) | np.isnan(sd), 1.0, sd)


def load_cohort(
    built: str | Path,
    seed: int = SEED,
    top_complaints: int = TOP_COMPLAINTS,
) -> Cohort:
    built = Path(built)
    ed = _load_visits(built)
    ed, raw, R = _attach_vitals(ed, built)
    ed["role"] = _split_patients(ed, seed)

    old = ed[ed["role"] == "old"]
    new = ed[ed["role"] == "new"]
    complaints = pd.concat([old, new])["complaint"].value_counts().index[:top_complaints].tolist()

    age_mean, age_sd = float(old["age"].mean()), float(old["age"].std(ddof=1))
    vit_mean = np.nanmean(raw[new.index.to_numpy()], axis=0)
    vit_sd = _safe_sd(raw[new.index.to_numpy()])

    splits: dict[str, Split] = {}
    base_names: list[str] = []
    vital_names: list[str] = []
    for name in SPLIT:
        idx = np.flatnonzero(ed["role"].to_numpy() == name)
        d = ed.iloc[idx].reset_index(drop=True)
        Xb, base_names = _base_design(d, complaints, age_mean, age_sd)
        V, vital_names = _vital_design(raw[idx], vit_mean, vit_sd)
        splits[name] = Split(
            visits=d,
            X_base=Xb,
            V=V,
            R=R[idx],
            y=d["died30"].to_numpy(dtype=float),
        )

    scaling = {
        "age": {"mean": age_mean, "sd": age_sd},
        "vitals": {
            name: {"mean": float(vit_mean[j]), "sd": float(vit_sd[j])}
            for j, name in enumerate(VITALS)
        },
    }
    return Cohort(
        splits=splits,
        base_feature_names=base_names,
        vital_feature_names=vital_names,
        vital_names=list(VITALS),
        complaints=complaints,
        scaling=scaling,
    )


def assert_disjoint(cohort: Cohort) -> None:
    seen: set[int] = set()
    for name, split in cohort.splits.items():
        patients = set(split.visits["lopnr_swed"].astype("int64"))
        overlap = seen & patients
        if overlap:
            raise AssertionError(f"{name} shares {len(overlap)} patients with an earlier split")
        seen |= patients


def summarise(cohort: Cohort, min_cell: int = MIN_CELL) -> pd.DataFrame:
    rows = []
    for name, split in cohort.splits.items():
        n = len(split.y)
        deaths = int(split.y.sum())
        rows.append(
            {
                "split": name,
                "visits": round5(n),
                "patients": round5(split.visits["lopnr_swed"].nunique()),
                "deaths": round5(deaths) if deaths >= min_cell else None,
                "prevalence": round(deaths / n, 4) if n else None,
                "hospitals": split.visits["ed_hospital"].nunique(),
                "vital_observed_pct": {
                    v: round(float(split.R[:, j].mean()), 3)
                    for j, v in enumerate(cohort.vital_names)
                },
            }
        )
    return pd.DataFrame(rows)


def save_split_manifest(
    cohort: Cohort,
    path: str | Path,
    built: str | Path,
    seed: int = SEED,
    min_cell: int = MIN_CELL,
) -> None:
    assert_disjoint(cohort)
    payload = {
        "built": str(built),
        "seed": seed,
        "split": SPLIT,
        "top_complaints": cohort.complaints,
        "scaling": cohort.scaling,
        "counts_rounded_to": 5,
        "min_cell": min_cell,
        "splits": {},
    }
    for name, split in cohort.splits.items():
        n, deaths = len(split.y), int(split.y.sum())
        payload["splits"][name] = {
            "visits": round5(n),
            "patients": round5(split.visits["lopnr_swed"].nunique()),
            "deaths": round5(deaths) if deaths >= min_cell else None,
            "prevalence": round(deaths / n, 4) if n else None,
            "vital_observed_pct": {
                v: round(float(split.R[:, j].mean()), 3)
                for j, v in enumerate(cohort.vital_names)
            },
        }
    Path(path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Summarise the built cohort and write the split manifest")
    p.add_argument("--built", type=Path, default=Path("data_pipeline/synthetic/built_synth_subset_50k"))
    p.add_argument("--out", type=Path, default=Path("evaluation/results/cohort_preview"))
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    cohort = load_cohort(args.built, seed=args.seed)
    assert_disjoint(cohort)
    args.out.mkdir(parents=True, exist_ok=True)
    summarise(cohort).to_csv(args.out / "cohort_summary.csv", index=False)
    save_split_manifest(cohort, args.out / "split_manifest.json", args.built, seed=args.seed)
    print(summarise(cohort).to_string(index=False))
