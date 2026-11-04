"""Do vital signs add to ed_atlas? 30-day mortality on the built data (step 05).

Three models, all for death within 30 days of an ED arrival:

    A  ed_atlas as is            P = E[sigmoid(z)], z ~ N(mu, sigma^2) from the atlas posterior
    B  recalibrated atlas        logit P = a + b * logit(P_A)
    C  recalibrated atlas + vitals (first value per visit; hinge terms around a normal value,
       plus a missing indicator per vital)

B and C are logistic regressions with a weak ridge (a Gaussian prior, as in
demo_model_update §3.2), evaluated with 5-fold cross-validation grouped by patient.
Needs only numpy and pandas, plus atlas_mortality_posterior.npz (300 KB) from ed_atlas.zip.

Writes aggregated CSVs only, with the same rules as 03_export_stats.py
(>= 10 patients per group, counts rounded to 5), so the output can go through the
normal export review.

    python 05_atlas_vitals.py                               # built_subset, atlas npz in this folder
    python 05_atlas_vitals.py --built .../built_clean_parquet --atlas path/to/npz

Caveat: the atlas was trained on SwED, probably on these same visits, so model A's
scores are optimistic. The comparison that matters is C against B.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
MIN_CELL, ROUND_TO = 10, 5
AGE_MIN, AGE_MAX = 18, 109
FOLDS, RIDGE, SEED = 5, 1.0, 0

# vital -> (concepts it is taken from, normal value, use values above normal, use values below normal)
VITALS = {
    "sbp": (["systolic_bp"], 120.0, True, True),
    "pulse": (["pulse_rate", "heart_rate"], 75.0, True, True),
    "resp_rate": (["resp_rate"], 16.0, True, True),
    "spo2": (["spo2"], 97.0, False, True),
    "temp": (["body_temp"], 37.0, True, True),
    "rls": (["consciousness"], 1.0, True, False),
}

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(64)
_GH_W = _GH_W / _GH_W.sum()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def round5(n) -> int:
    return int(ROUND_TO * round(n / ROUND_TO))


# =============================================================================
# ed_atlas mortality model (numpy-only port of ed_atlas.ParameterPosterior.predict)
# =============================================================================

class AtlasMortality:
    def __init__(self, npz: Path):
        d = np.load(npz)
        self.beta, self.base_cov = d["beta_mean"].astype(float), d["base_cov"].astype(float)
        self.sym_covs, self.cross_covs = d["sym_covs"].astype(float), d["cross_covs"].astype(float)
        self.cr = d["cr_basis"].astype(float)
        self.sctids = d["sctids"].tolist()
        self.index = {s: i for i, s in enumerate(self.sctids)}

    def _dims(self, si: np.ndarray) -> np.ndarray:
        S = len(self.sctids)
        cols = [10 + si, 10 + S + si] + [10 + 2 * S + 4 * si + k for k in range(4)] \
            + [10 + 6 * S + 4 * si + k for k in range(4)]
        return np.stack(cols, axis=1)  # (n, 10)

    def mu_sigma(self, age: np.ndarray, male: np.ndarray, sctid: list[str]) -> tuple[np.ndarray, np.ndarray]:
        si = np.array([self.index[s] for s in sctid])
        cr = self.cr[age - AGE_MIN]
        x = np.column_stack([np.ones(len(age)), male, cr, male[:, None] * cr])  # same for base and symptom
        mu = x @ self.beta[:10] + (x * self.beta[self._dims(si)]).sum(1)
        var = np.einsum("ni,ij,nj->n", x, self.base_cov, x) \
            + np.einsum("ni,nij,nj->n", x, self.sym_covs[si], x) \
            + 2 * np.einsum("ni,nij,nj->n", x, self.cross_covs[si], x)
        return mu, np.sqrt(np.maximum(var, 0))

    def p_death(self, age, male, sctid) -> np.ndarray:
        mu, sigma = self.mu_sigma(age, male, sctid)
        return sigmoid(mu[:, None] + sigma[:, None] * _GH_X) @ _GH_W


# =============================================================================
# Data
# =============================================================================

def load_visits(built: Path, atlas: AtlasMortality, counts: dict) -> pd.DataFrame:
    ed = pd.read_parquet(built / "ed_visits.parquet", columns=[
        "ed_visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_age", "is_male",
        "ed_presented_complaint_std"])
    counts["visits in build"] = len(ed)
    ok = ed["ed_age"].between(AGE_MIN, AGE_MAX) & ed["is_male"].notna()
    counts["visits dropped: age outside 18-109 or sex missing"] = int((~ok).sum())
    ed = ed[ok].copy()
    ed["age"] = ed["ed_age"].astype(int)
    ed["male"] = ed["is_male"].astype(float)

    # Up to two SNOMED codes per visit; unknown or missing complaints use the atlas's "other" category.
    codes = ed["ed_presented_complaint_std"].fillna("").map(lambda s: re.findall(r"\[(\d+)\]", s))
    known = codes.map(lambda c: [x for x in c if x in atlas.index][:2])
    ed["complaint_in_atlas"] = known.map(len) > 0
    ed["sct1"] = known.map(lambda c: c[0] if c else "nosct:other")
    ed["sct2"] = known.map(lambda c: c[1] if len(c) > 1 else None)
    p1 = atlas.p_death(ed["age"].to_numpy(), ed["male"].to_numpy(), ed["sct1"].tolist())
    two = ed["sct2"].notna().to_numpy()
    p = p1.copy()
    if two.any():
        p2 = atlas.p_death(ed["age"].to_numpy()[two], ed["male"].to_numpy()[two], ed.loc[two, "sct2"].tolist())
        p[two] = 0.5 * p1[two] + 0.5 * p2
    ed["p_atlas"] = p

    deaths = pd.read_parquet(built / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
    ed = ed.merge(deaths, on="lopnr_swed", how="left")
    arrival_day = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    days = (pd.to_datetime(ed["date_death_minimp"]) - arrival_day).dt.days
    before = days < 0
    counts["visits dropped: death date before arrival"] = int(before.sum())
    ed = ed[~before.to_numpy()].copy()
    ed["died30"] = (days[~before] <= 30).fillna(False).astype(float).to_numpy()
    return ed


def add_vitals(ed: pd.DataFrame, built: Path) -> tuple[pd.DataFrame, list[str]]:
    v = pd.read_parquet(built / "vitals.parquet", columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"])
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"]]
    first = (v.sort_values("datetime_analysis").groupby(["ed_visit_id", "concept"])["value"].first()
             .unstack("concept"))
    cols = []
    for name, (concepts, normal, above, below) in VITALS.items():
        have = [c for c in concepts if c in first]
        val = first[have].bfill(axis=1).iloc[:, 0] if have else pd.Series(dtype=float)
        val = ed["ed_visit_id"].map(val)
        ed[f"{name}_missing"] = val.isna().astype(float)
        cols.append(f"{name}_missing")
        if above:
            ed[f"{name}_above"] = (val - normal).clip(lower=0).fillna(0)
            cols.append(f"{name}_above")
        if below:
            ed[f"{name}_below"] = (normal - val).clip(lower=0).fillna(0)
            cols.append(f"{name}_below")
    ed["any_vital"] = (ed[[f"{n}_missing" for n in VITALS]].min(axis=1) == 0)
    return ed, cols


# =============================================================================
# Models
# =============================================================================

def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray]:
    """MAP logistic regression with a N(0, 1/ridge) prior on every coefficient but the intercept."""
    pen = np.r_[0.0, np.full(X.shape[1] - 1, ridge)]
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = sigmoid(X @ b)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (y - p) - pen * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b, np.linalg.inv(H)


def design(ed: pd.DataFrame, feats: list[str], mean=None, sd=None):
    Z = ed[feats].to_numpy(float)
    mean = Z.mean(0) if mean is None else mean
    sd = Z.std(0) if sd is None else sd
    sd = np.where(sd > 0, sd, 1.0)
    return np.column_stack([np.ones(len(Z)), (Z - mean) / sd]), mean, sd


def cross_validate(ed: pd.DataFrame, feats: list[str]) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    pts = ed["lopnr_swed"].unique()
    fold_of = dict(zip(pts, rng.integers(0, FOLDS, len(pts))))
    fold = ed["lopnr_swed"].map(fold_of).to_numpy()
    y, out = ed["died30"].to_numpy(), np.empty(len(ed))
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        Xtr, m, s = design(ed[tr], feats)
        b, _ = fit_logistic(Xtr, y[tr])
        out[te] = sigmoid(design(ed[te], feats, m, s)[0] @ b)
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def scores(y, p) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    slope_int, _ = fit_logistic(np.column_stack([np.ones(len(p)), logit(p)]), y, ridge=0.0)
    return {"auc": round(auc(y, p), 4), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
            "mean_predicted": round(float(p.mean()), 4), "observed_rate": round(float(y.mean()), 4),
            "calibration_intercept": round(float(slope_int[0]), 3), "calibration_slope": round(float(slope_int[1]), 3)}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, default=OUT_ROOT / "built_subset", help="output folder of 02_build.py")
    ap.add_argument("--atlas", type=Path, default=Path("atlas_mortality_posterior.npz"))
    ap.add_argument("--out", type=Path, help="default: $SWED_OUT/atlas_vitals_<built folder name>")
    args = ap.parse_args()
    out = args.out or OUT_ROOT / f"atlas_vitals_{args.built.name}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"built: {args.built}\n           atlas: {args.atlas}\n           out:   {out}")

    atlas = AtlasMortality(args.atlas)
    counts: dict = {}
    ed = load_visits(args.built, atlas, counts)
    ed, vital_cols = add_vitals(ed, args.built)
    y = ed["died30"].to_numpy()
    log(f"{len(ed):,} visits, {ed['lopnr_swed'].nunique():,} patients, {int(y.sum()):,} deaths within 30 days")
    if y.sum() < MIN_CELL:
        raise SystemExit("fewer than 10 deaths within 30 days: too few to fit or to export")

    ed["atlas_logit"] = logit(ed["p_atlas"].to_numpy())
    preds = {"A_atlas": ed["p_atlas"].to_numpy(),
             "B_atlas_recalibrated": cross_validate(ed, ["atlas_logit"]),
             "C_atlas_plus_vitals": cross_validate(ed, ["atlas_logit"] + vital_cols)}

    # --- overall and per-hospital scores (out-of-fold for B and C) ---
    rows = []
    groups = [("all", np.ones(len(ed), bool))] + [(h, (ed["ed_hospital"] == h).to_numpy()) for h in sorted(ed["ed_hospital"].dropna().unique())]
    for g, m in groups:
        persons, deaths = ed.loc[m, "lopnr_swed"].nunique(), y[m].sum()
        if persons < MIN_CELL or deaths < MIN_CELL:
            continue
        for name, p in preds.items():
            rows.append({"group": g, "model": name, "n": round5(m.sum()), "persons": round5(persons), **scores(y[m], p[m])})
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    # --- calibration by decile of predicted risk ---
    cal = []
    for name, p in preds.items():
        dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        for d in range(10):
            m = (dec == d).to_numpy()
            if ed.loc[m, "lopnr_swed"].nunique() >= MIN_CELL:
                cal.append({"model": name, "decile": d + 1, "n": round5(m.sum()),
                            "mean_predicted": round(float(p[m].mean()), 4), "observed_rate": round(float(y[m].mean()), 4)})
    pd.DataFrame(cal).to_csv(out / "calibration.csv", index=False)

    # --- coefficients of model C on all data (per SD of each feature) ---
    feats = ["atlas_logit"] + vital_cols
    X, _, sd = design(ed, feats)
    b, cov = fit_logistic(X, y)
    se = np.sqrt(np.diag(cov))
    coef = pd.DataFrame({"term": ["intercept"] + feats, "sd_of_feature": np.r_[np.nan, sd].round(3),
                         "coef_per_sd": b.round(4), "se": se.round(4),
                         "odds_ratio_per_sd": np.exp(b).round(3),
                         "or_low95": np.exp(b - 1.96 * se).round(3), "or_high95": np.exp(b + 1.96 * se).round(3)})
    coef.to_csv(out / "coefficients_C.csv", index=False)

    # --- coverage ---
    cov_rows = [{"item": k, "value": round5(v)} for k, v in counts.items()]
    for h, m in groups:
        sub = ed[m]
        if sub["lopnr_swed"].nunique() >= MIN_CELL:
            cov_rows += [{"item": f"{h}: visits used", "value": round5(len(sub))},
                         {"item": f"{h}: share with complaint in atlas", "value": round(float(sub["complaint_in_atlas"].mean()), 3)},
                         {"item": f"{h}: share with any vital", "value": round(float(sub["any_vital"].mean()), 3)},
                         {"item": f"{h}: share died within 30 days", "value": round(float(sub["died30"].mean()), 4)}]
    pd.DataFrame(cov_rows).to_csv(out / "coverage.csv", index=False)

    s = pd.DataFrame(rows)
    log("scores (all visits):\n" + s[s["group"] == "all"][["model", "auc", "brier", "calibration_slope"]].to_string(index=False))
    log(f"done. Read every CSV in {out} before export.")


if __name__ == "__main__":
    main()
