#!/usr/bin/env python3
"""Load exported joint/conditional summaries and sample linked synthetic values."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

CONDITION_KEYS = ["age_band", "sex", "complaint", "outcome"]


def _read(stats_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(stats_dir) / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def _lookup(df: pd.DataFrame, **keys) -> pd.DataFrame:
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        if k not in df:
            return df.iloc[0:0]
        mask &= df[k].fillna("all").astype(str).eq(str(v)).to_numpy()
    return df[mask]


def context_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    age = pd.to_numeric(frame["ed_age"], errors="coerce") if "ed_age" in frame else pd.Series(np.nan, index=frame.index)
    out["age_band"] = "all"
    out.loc[age < 40, "age_band"] = "18_39"
    out.loc[(age >= 40) & (age < 60), "age_band"] = "40_59"
    out.loc[(age >= 60) & (age < 80), "age_band"] = "60_79"
    out.loc[age >= 80, "age_band"] = "80+"
    sex = frame["is_male"] if "is_male" in frame else pd.Series(pd.NA, index=frame.index)
    out["sex"] = sex.astype("Int64").astype("string").fillna("all").astype(str)
    complaint = frame["ed_presented_complaint_std"] if "ed_presented_complaint_std" in frame else pd.Series(pd.NA, index=frame.index)
    code = complaint.astype("string").str.extract(r"\[(\d+)\]", expand=False)
    out["complaint"] = code.fillna("all").astype(str)
    died = frame["died30"] if "died30" in frame else pd.Series(pd.NA, index=frame.index)
    out["outcome"] = pd.to_numeric(died, errors="coerce").astype("Int64").astype("string").fillna("all").astype(str)
    return out


def _psd(rho: np.ndarray) -> np.ndarray:
    rho = np.nan_to_num(np.asarray(rho, dtype=float), nan=0.0)
    rho = (rho + rho.T) / 2
    np.fill_diagonal(rho, 1.0)
    w, v = np.linalg.eigh(rho)
    out = (v * np.maximum(w, 1e-6)) @ v.T
    scale = np.sqrt(np.diag(out))
    out = out / np.maximum(scale, 1e-12)[:, None] / np.maximum(scale, 1e-12)[None, :]
    np.fill_diagonal(out, 1.0)
    return out


class JointStats:
    def __init__(self, stats_dir: str | Path):
        self.stats_dir = Path(stats_dir)
        self.correlation = _read(stats_dir, "joint_correlation")
        self.conditional = _read(stats_dir, "conditional_quantiles")
        self.categorical = _read(stats_dir, "joint_categorical")
        self.mortality = _read(stats_dir, "mortality_rate")
        self.diagnoses = _read(stats_dir, "diagnoses_conditional")
        self.monthly = _read(stats_dir, "ed_visits_hospital_month")
        self.intervals = _read(stats_dir, "ed_visit_intervals")
        for df, numeric in ((self.correlation, ("rho", "n", "persons")),
                            (self.conditional, ("n", "persons")),
                            (self.categorical, ("n", "persons")),
                            (self.mortality, ("n", "persons", "died30_mean", "died30_rate")),
                            (self.diagnoses, ("n", "persons")),
                            (self.monthly, ("n", "persons")),
                            (self.intervals, ("n", "persons"))):
            if df.empty:
                continue
            for c in numeric:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        if not self.conditional.empty:
            for c in self.conditional.columns:
                if c == "null_rate" or (c.startswith("q") and c[1:].isdigit()):
                    self.conditional[c] = pd.to_numeric(self.conditional[c], errors="coerce")

    @property
    def available(self) -> bool:
        return any(not df.empty for df in (self.correlation, self.conditional, self.categorical,
                                           self.mortality, self.diagnoses, self.monthly, self.intervals))

    def correlation_matrix(self, domain: str, group: str, source: str, variables: list[str]) -> np.ndarray | None:
        if self.correlation.empty:
            return None
        df = self.correlation
        rows = _lookup(df, domain=domain, source=source)
        for candidate in [group, "all"]:
            found = rows[rows["group"].astype(str) == str(candidate)]
            if not found.empty:
                rows = found
                break
        matrix = np.eye(len(variables))
        found = False
        for i, a in enumerate(variables):
            for j, b in enumerate(variables):
                if i >= j:
                    continue
                pair = rows[((rows["var_a"] == a) & (rows["var_b"] == b))
                            | ((rows["var_a"] == b) & (rows["var_b"] == a))]
                if not pair.empty and pd.notna(pair.iloc[0]["rho"]):
                    matrix[i, j] = matrix[j, i] = float(pair.iloc[0]["rho"])
                    found = True
        return _psd(matrix) if found else None

    def uniforms(self, domain: str, group: str, source: str, variables: list[str], n: int,
                 rng: np.random.Generator) -> np.ndarray:
        u = rng.random((n, len(variables)))
        rho = self.correlation_matrix(domain, group, source, variables)
        if rho is None or len(variables) < 2:
            return u
        z = rng.standard_normal((n, len(variables))) @ np.linalg.cholesky(rho).T
        return ndtr(z)

    def _condition_levels(self, group: str, source: str, variable: str,
                          ctx: dict[str, str]) -> list[dict[str, str]]:
        base = {"domain": "", "group": str(group), "source": str(source), "variable": str(variable)}
        levels = []
        for mask in [(1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)]:
            level = dict(base)
            for key, keep in zip(CONDITION_KEYS, mask):
                level[key] = ctx.get(key, "all") if keep else "all"
            levels.append(level)
        for mask in [(1, 1, 1, 1), (1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 0, 0)]:
            level = dict(base, group="all")
            for key, keep in zip(CONDITION_KEYS, mask):
                level[key] = ctx.get(key, "all") if keep else "all"
            levels.append(level)
        levels.append(dict(base, group="all", age_band="all", sex="all", complaint="all", outcome="all", domain=""))
        return levels

    def conditional_row(self, domain: str, group: str, source: str, variable: str,
                        ctx: dict[str, str]) -> pd.Series | None:
        if self.conditional.empty:
            return None
        for level in self._condition_levels(group, source, variable, ctx):
            level["domain"] = domain
            found = _lookup(self.conditional, **level)
            if not found.empty and pd.notna(found.iloc[0].get("q50")):
                return found.iloc[0]
        return None

    def categorical_options(self, hospital: str, column_a: str, value_a: str, column_b: str) -> pd.DataFrame:
        for h in (str(hospital), "all"):
            found = _lookup(self.categorical, ed_hospital=h, column_a=column_a,
                            value_a=str(value_a), column_b=column_b)
            if not found.empty:
                return found
        return pd.DataFrame()

    def mortality_rate(self, group: str, age_band: str, sex: str, complaint: str, default: float) -> float:
        if self.mortality.empty:
            return float(default)
        levels = [
            {"group": group, "age_band": age_band, "sex": sex, "complaint": complaint},
            {"group": group, "age_band": age_band, "sex": sex, "complaint": "all"},
            {"group": group, "age_band": age_band, "sex": "all", "complaint": "all"},
            {"group": group, "age_band": age_band, "sex": "all", "complaint": "all"},
            {"group": group, "age_band": "all", "sex": "all", "complaint": "all"},
            {"group": "all", "age_band": age_band, "sex": sex, "complaint": complaint},
            {"group": "all", "age_band": age_band, "sex": sex, "complaint": "all"},
            {"group": "all", "age_band": age_band, "sex": "all", "complaint": "all"},
            {"group": "all", "age_band": "all", "sex": "all", "complaint": "all"},
        ]
        for level in levels:
            found = _lookup(self.mortality, **level)
            if not found.empty and pd.notna(found.iloc[0].get("died30_rate")):
                return float(found.iloc[0]["died30_rate"])
        return float(default)

    def diagnosis_options(self, age_band: str, sex: str, complaint: str, outcome: str) -> pd.DataFrame:
        levels = [
            {"age_band": age_band, "sex": sex, "complaint": complaint, "outcome": outcome},
            {"age_band": age_band, "sex": sex, "complaint": complaint, "outcome": "all"},
            {"age_band": age_band, "sex": sex, "complaint": "all", "outcome": "all"},
            {"age_band": age_band, "sex": "all", "complaint": "all", "outcome": "all"},
            {"age_band": "all", "sex": "all", "complaint": "all", "outcome": "all"},
        ]
        for level in levels:
            found = _lookup(self.diagnoses, **level)
            if not found.empty:
                return found
        return pd.DataFrame()

    def month_weights(self, hospital: str) -> pd.DataFrame:
        found = _lookup(self.monthly, ed_hospital=hospital)
        if found.empty:
            found = _lookup(self.monthly, ed_hospital="all")
        return found

    def interval_row(self, hospital: str) -> pd.Series | None:
        for h in (str(hospital), "all"):
            found = _lookup(self.intervals, ed_hospital=h)
            if not found.empty and pd.notna(found.iloc[0].get("q50")):
                return found.iloc[0]
        return None
