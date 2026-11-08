#!/usr/bin/env python3
"""Updating ed_atlas with vital signs: 30-day mortality on the built data (step 08).

Step 1 of the build plan is the data: one row per ED visit with the outcome, the vital-sign
features and a patient-level split. Step 2 adds the models (sections 5-7: the atlas, one Laplace
fitting engine, and recal / new_only / joint_v2), checked on simulated data in
synthetic/sim_update_atlas.py. Step 3 wires them into main with twins, scores and the bootstrap.

    visits     ages 18-109, sex known, death date not before arrival
    outcome    died within 0-30 days of the arrival day (date_death_minimp)
    vitals     first plausible value per visit, measured from arrival - 60 min (02's window)
               to arrival + --cutoff minutes (default 60, the triage-time placeholder)
    features   per continuous vital: z, z^2, missing flag; RLS as categories 2, 3, 4+ (vs 1) + missing flag
    split      by patient: 20% test, then 10% of the rest as validation (picks tau), the rest train

Writes aggregated CSVs only, with the rules of 03_export_stats.py (>= 10 patients per group,
counts rounded to 5):

    cohort.csv           visits, patients and deaths per role (train / val / test)
    coverage_curve.csv   share of visits with a first value by each minute after arrival, per vital
                         and hospital: the data behind the choice of --cutoff
    vitals_summary.csv   share present, mean and SD of each vital in the update set (train + val)

    python 08_update_atlas.py                                   # built_subset, cutoff 60 min
    python 08_update_atlas.py --built ../data/built_synth_subset_50k --out ../data/update_atlas_synth
    python 08_update_atlas.py --years-from 2016                 # recent visits only (atlas overlap check)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
MIN_CELL, ROUND_TO = 10, 5
AGE_MIN, AGE_MAX = 18, 109  # the atlas's age grid
SEED = 42
TEST_SHARE, VAL_SHARE = 0.2, 0.1  # test: of all patients; validation: of the remaining update patients

# vital -> concepts in vitals.parquet it is taken from (the earliest of them wins)
CONTINUOUS = {
    "sbp": ["systolic_bp"],
    "pulse": ["pulse_rate", "heart_rate"],
    "resp_rate": ["resp_rate"],
    "spo2": ["spo2"],
    "temp": ["body_temp"],
}
RLS = ["consciousness"]
RLS_LEVELS = {"rls_2": (2, 2), "rls_3": (3, 3), "rls_4plus": (4, 8)}  # reference: RLS 1 (alert)
VITALS = {**CONTINUOUS, "rls": RLS}

# minutes after arrival at which coverage is reported; "end_of_stay" = 02's whole window
CURVE_MINUTES = [-60, -30, 0, 15, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def round5(n) -> int:
    return int(ROUND_TO * round(n / ROUND_TO))


# =============================================================================
# 1. Visits and outcome
# =============================================================================

def load_visits(built: Path, years_from: int | None, counts: dict) -> pd.DataFrame:
    """One row per visit, with the outcome
     * same filters as 07: the atlas's age range, sex known, death date not before arrival
     * complaint = the first SNOMED code in ed_presented_complaint_std ("Bröstsmärta [29857009]"), else None
     * dead = 1 if the death date is 0-30 days after the arrival day (Stockholm time); later visits
       of a patient who dies count too, so one death can make several visits positive
    """
    ed = pd.read_parquet(built / "ed_visits.parquet", columns=[
        "ed_visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_age", "is_male",
        "ed_presented_complaint_std"])
    counts["visits in build"] = len(ed)
    ok = ed["ed_age"].between(AGE_MIN, AGE_MAX) & ed["is_male"].notna() & ed["ed_arrival_datetime"].notna()
    counts["visits dropped: age outside 18-109, sex or arrival missing"] = int((~ok).sum())
    ed = ed[ok].copy()
    if years_from is not None:
        recent = ed["ed_arrival_datetime"].dt.year >= years_from
        counts[f"visits dropped: arrival before {years_from}"] = int((~recent).sum())
        ed = ed[recent].copy()

    ed["age"] = ed["ed_age"].astype(int)
    ed["male"] = ed["is_male"].astype(int)
    ed["complaint"] = ed["ed_presented_complaint_std"].str.extract(r"\[(\d+)\]", expand=False)

    deaths = pd.read_parquet(built / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
    death = pd.to_datetime(ed["lopnr_swed"].map(deaths.set_index("lopnr_swed")["date_death_minimp"]))
    arrival_day = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    days = (death - arrival_day).dt.days
    before = days < 0  # death before the visit: a data error
    counts["visits dropped: death date before arrival"] = int(before.sum())
    ed = ed[~before].copy()
    ed["dead"] = days[~before].between(0, 30).astype(int)  # NaN (no death) -> False
    return ed.drop(columns=["ed_age", "is_male", "ed_presented_complaint_std"]).reset_index(drop=True)


# =============================================================================
# 2. Vital signs: time of the first value, and the first value within the cutoff
# =============================================================================

def first_vitals(ed: pd.DataFrame, built: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per visit and vital: the earliest plausible value and its time in minutes after arrival
     * 02 already linked each measurement to the visit whose window holds it (arrival - 60 min
       to the end of the stay), so every row here is inside that window
     * implausible values (outside 02's PLAUSIBLE ranges) are dropped, not used
     * pulse: the earliest of pulse_rate and heart_rate
     * several values at the earliest time (double entry, two devices): their mean, so the
       result doesn't depend on row order
     * returns (value, minutes): two visits x vitals tables, NaN where the visit has no value
    """
    v = pd.read_parquet(built / "vitals.parquet", columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"])
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & v["datetime_analysis"].notna() & ~v["implausible"]]
    to_vital = {c: name for name, concepts in VITALS.items() for c in concepts}
    v = v[v["concept"].isin(to_vital)].assign(vital=lambda d: d["concept"].map(to_vital),
                                              ed_visit_id=lambda d: d["ed_visit_id"].astype("int64"))
    arrival = ed.set_index("ed_visit_id")["ed_arrival_datetime"]
    v = v[v["ed_visit_id"].isin(arrival.index)]
    v["minutes"] = (v["datetime_analysis"] - v["ed_visit_id"].map(arrival)).dt.total_seconds() / 60

    earliest = v.groupby(["ed_visit_id", "vital"])["minutes"].transform("min")
    first = v[v["minutes"] == earliest].groupby(["ed_visit_id", "vital"]).agg(value=("value", "mean"), minutes=("minutes", "first"))
    value = first["value"].unstack().reindex(index=ed["ed_visit_id"], columns=list(VITALS))
    minutes = first["minutes"].unstack().reindex(index=ed["ed_visit_id"], columns=list(VITALS))
    return value.set_axis(ed.index), minutes.set_axis(ed.index)


def coverage_curve(ed: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Share of visits whose first value of a vital is at or before each minute after arrival
     * one curve per vital and for "any" vital, for all visits and per hospital (>= 10 patients)
     * a steep rise that then flattens marks triage-time measurement; the flat part is where
       --cutoff belongs (Q23: the cutoff should come from this curve, not from a round number)
    """
    t = minutes.assign(any=minutes.min(axis=1))
    groups = [("all", np.ones(len(ed), bool))] + [
        (h, (ed["ed_hospital"] == h).to_numpy()) for h in sorted(ed["ed_hospital"].dropna().unique())]
    rows = []
    for g, m in groups:
        if ed.loc[m, "lopnr_swed"].nunique() < MIN_CELL:
            continue
        for vital in t.columns:
            col = t.loc[m, vital]
            for by in CURVE_MINUTES + ["end_of_stay"]:
                hit = col.notna() if by == "end_of_stay" else col <= by
                rows.append({"group": g, "vital": vital, "by_minute": by, "visits": round5(m.sum()),
                             "share": round(float(hit.mean()), 4)})
    return pd.DataFrame(rows)


# =============================================================================
# 3. Split by patient
# =============================================================================

def split_patients(ed: pd.DataFrame, seed: int) -> pd.Series:
    """Role per visit: train, val or test
     * patients are shuffled and every patient keeps all their visits in one role, so no
       patient is both fitted on and scored on
     * test = TEST_SHARE of the patients; val = VAL_SHARE of the rest (used to pick tau);
       train = the rest. train + val together are the "update set"
    """
    rng = np.random.default_rng(seed)
    patients = rng.permutation(ed["lopnr_swed"].unique())
    n_test = int(round(TEST_SHARE * len(patients)))
    n_val = int(round(VAL_SHARE * (len(patients) - n_test)))
    role = pd.Series("train", index=patients)
    role.iloc[:n_test] = "test"
    role.iloc[n_test:n_test + n_val] = "val"
    return ed["lopnr_swed"].map(role)


# =============================================================================
# 4. Features
# =============================================================================

def vital_scale(value: pd.DataFrame, update: np.ndarray) -> pd.DataFrame:
    """Mean and SD of each continuous vital, from the update set only (train + val)
     * test never sets a scale, so nothing about the test patients leaks into the features
     * one scale for every model and every learning-curve subsample, so the features of a
       visit are the same whichever model or sample size is fitted
     * NaN skipped: only visits that have the vital count
    """
    u = value.loc[update, list(CONTINUOUS)]
    return pd.DataFrame({"mean": u.mean(), "sd": u.std(), "share_present": u.notna().mean()})


def vital_features(value: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    """Vital features: 3 columns per continuous vital, 4 for RLS (5 * 3 + 4 = 19)
     * <v>_z: standardised value, 0 when missing (the update-set average)
     * <v>_z2: its square, so both low and high values can raise the risk (U shape)
     * <v>_missing: 1 when the visit has no value within the cutoff
     * a missing vital adds only its flag's coefficient, since z = z^2 = 0
     * RLS is ordinal (1 alert ... 8 unresponsive), so categories instead of a parabola:
       rls_2, rls_3, rls_4plus vs RLS 1; 4-8 are merged because they are rare
    """
    X = {}
    for name in CONTINUOUS:
        z = (value[name] - scale.loc[name, "mean"]) / scale.loc[name, "sd"]
        X[f"{name}_z"] = z.fillna(0)
        X[f"{name}_z2"] = (z ** 2).fillna(0)
        X[f"{name}_missing"] = value[name].isna().astype(float)
    rls = value["rls"].round()
    for col, (lo, hi) in RLS_LEVELS.items():
        X[col] = rls.between(lo, hi).astype(float)  # NaN -> 0
    X["rls_missing"] = rls.isna().astype(float)
    return pd.DataFrame(X, index=value.index)


# =============================================================================
# 5. ed_atlas, one complaint at a time
# =============================================================================

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(64)
_GH_W = _GH_W / _GH_W.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expected_sigmoid(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """E[sigmoid(eta)] for eta ~ N(mean, var), by 64-point Gauss-Hermite
     * the risk averaged over the coefficients' uncertainty, as the atlas package defines it (Q4)
     * every model here predicts this way, so they are compared like for like
    """
    return sigmoid(mean[:, None] + np.sqrt(np.maximum(var, 0))[:, None] * _GH_X) @ _GH_W


class Atlas:
    """ed_atlas mortality model, with one coefficient vector per complaint (Q24 option a)
     * logit P = x . phi_c, x = (1, male, cr(age) [4], male * cr(age) [4]): 10 numbers per visit
     * phi_c = the base coefficients + complaint c's offsets, so the atlas posterior for one
       complaint is N(mean[c], cov[c]) with cov[c] = base + sym_c + cross_c + cross_c^T, built from
       exactly the blocks the npz stores
     * the npz has no complaint x complaint blocks, so complaints are independent under this prior
     * the 4 spline columns sum to 1, the same as the intercept: only the age curve is identified,
       and the atlas posterior keeps its prior variance along that one direction (harmless for risks)
    """

    def __init__(self, npz: Path):
        d = np.load(npz)
        beta = d["beta_mean"].astype(float)
        self.sctids = [str(s) for s in d["sctids"]]
        self.index = {s: i for i, s in enumerate(self.sctids)}
        self.cr = d["cr_basis"].astype(float)
        S, i = len(self.sctids), np.arange(len(self.sctids))
        dims = np.stack([10 + i, 10 + S + i] + [10 + 2 * S + 4 * i + k for k in range(4)]
                        + [10 + 6 * S + 4 * i + k for k in range(4)], axis=1)  # (S, 10): complaint c's columns
        self.mean = beta[:10] + beta[dims]  # (S, 10)
        cross = d["cross_covs"].astype(float)
        self.cov = d["base_cov"].astype(float) + d["sym_covs"].astype(float) + cross + cross.transpose(0, 2, 1)

    def design(self, age: np.ndarray, male: np.ndarray) -> np.ndarray:
        cr = self.cr[np.asarray(age) - AGE_MIN]
        male = np.asarray(male, float)
        return np.column_stack([np.ones(len(cr)), male, cr, male[:, None] * cr])  # (n, 10)

    def complaint_index(self, complaint: pd.Series) -> np.ndarray:
        """Atlas row per visit; a complaint the atlas doesn't know (or none) -> "nosct:other"."""
        idx = complaint.map(self.index)
        if idx.isna().any():
            if "nosct:other" not in self.index:
                raise SystemExit(f"{int(idx.isna().sum())} visits have a complaint the atlas lacks, and it has no 'nosct:other'")
            idx = idx.fillna(self.index["nosct:other"])
        return idx.astype(int).to_numpy()

    def p_death(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        mean = (x * self.mean[g]).sum(1)
        var = np.einsum("ni,nij,nj->n", x, self.cov[g], x)
        return expected_sigmoid(mean, var)


# =============================================================================
# 6. Laplace fit of a grouped logistic regression
# =============================================================================

"""One fitting engine for all three update methods
 * logit P_i = Xg_i . phi_{g(i)} + V_i . gamma
     g(i)   the visit's group: its complaint (new_only, joint_v2) or one group for everyone (recal)
     phi_c  d coefficients per group, prior N(mg[c], Pg[c]^-1)
     gamma  k coefficients shared by all visits (the vitals), prior N(mv, Pv^-1)
 * Laplace: find the posterior's peak by Newton's method, then approximate the posterior by the
   Gaussian with that mean and covariance = inverse of minus the Hessian at the peak
 * a visit touches only its own group's phi and gamma, so the Hessian is block-diagonal in the
   groups plus a dense border for gamma (an "arrowhead"); solving it group by group keeps the
   cost linear in the number of complaints (the same sparsity the atlas itself exploits)
"""


@dataclass
class Posterior:
    phi: np.ndarray    # (G, d) posterior mean per group
    gamma: np.ndarray  # (k,) posterior mean of the shared coefficients
    S_gg: np.ndarray   # (G, d, d) covariance within each group
    S_gv: np.ndarray   # (G, d, k) covariance between each group and gamma
    S_vv: np.ndarray   # (k, k) covariance of gamma
    iterations: int

    def predict(self, g: np.ndarray, Xg: np.ndarray, V: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """Risk per visit, averaged over the Gaussian posterior: eta ~ N(m, s^2), P = E[sigmoid(eta)]."""
        out = np.empty(len(g))
        for a in range(0, len(g), chunk):
            s = slice(a, a + chunk)
            x, v, gs = Xg[s], V[s], g[s]
            m = (x * self.phi[gs]).sum(1) + v @ self.gamma
            var = (np.einsum("ni,nij,nj->n", x, self.S_gg[gs], x)
                   + 2 * np.einsum("ni,nik,nk->n", x, self.S_gv[gs], v)
                   + np.einsum("nk,kl,nl->n", v, self.S_vv, v))
            out[s] = expected_sigmoid(m, var)
        return out


def fit_laplace(g, Xg, V, y, mg, Pg, mv, Pv, max_iter: int = 100, tol: float = 1e-8) -> Posterior:
    """Laplace approximation of the posterior (see the block comment above)
     * g (n,) group index, Xg (n, d), V (n, k) (k may be 0), y (n,) 0/1
     * mg (G, d), Pg (G, d, d): prior mean and precision per group; mv (k,), Pv (k, k): for gamma
     * starts at the prior mean (for joint_v2: the atlas), Newton steps halved if they would lower
       the log posterior, stops when no coefficient moves more than tol
    """
    G, d = mg.shape
    k = V.shape[1]
    order = np.argsort(g, kind="stable")
    g, Xg, V, y = g[order], Xg[order], V[order], np.asarray(y, float)[order]
    bounds = np.searchsorted(g, np.arange(G + 1))  # rows of group c: bounds[c] to bounds[c + 1]

    def eta(phi, gamma):
        return (Xg * phi[g]).sum(1) + V @ gamma

    def log_post(phi, gamma, e):
        dg, dv = phi - mg, gamma - mv
        return ((y * e - np.logaddexp(0, e)).sum()
                - 0.5 * np.einsum("ci,cij,cj->", dg, Pg, dg) - 0.5 * dv @ Pv @ dv)

    def blocks(phi, gamma, e):
        """Gradient and minus-Hessian blocks: A (G, d, d) per group, B (G, d, k) group x gamma, D (k, k)."""
        p = sigmoid(e)
        w, r = p * (1 - p), y - p
        A, B = Pg.copy(), np.zeros((G, d, k))
        grad_g = -np.einsum("cij,cj->ci", Pg, phi - mg)
        for c in range(G):
            s = slice(bounds[c], bounds[c + 1])
            if s.start == s.stop:
                continue  # no visits: the group stays at its prior
            xw = Xg[s] * w[s, None]
            A[c] += xw.T @ Xg[s]
            B[c] = xw.T @ V[s]
            grad_g[c] += Xg[s].T @ r[s]
        D = Pv + (V * w[:, None]).T @ V
        grad_v = V.T @ r - Pv @ (gamma - mv)
        return A, B, D, grad_g, grad_v

    def solve(A, B, D, grad_g, grad_v):
        """Arrowhead solve: eliminate each group's block, solve the k x k Schur system for gamma, back-substitute."""
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B
        schur = D - np.einsum("cdk,cdl->kl", B, AinvB)
        step_v = np.linalg.solve(schur, grad_v - np.einsum("cdk,cd->k", AinvB, grad_g)) if k else np.zeros(0)
        step_g = np.einsum("cij,cj->ci", Ainv, grad_g - np.einsum("cdk,k->cd", B, step_v))
        return step_g, step_v, Ainv, AinvB, schur

    phi, gamma = mg.copy(), np.asarray(mv, float).copy()
    e = eta(phi, gamma)
    lp = log_post(phi, gamma, e)
    for it in range(1, max_iter + 1):
        step_g, step_v, *_ = solve(*blocks(phi, gamma, e))
        t = 1.0
        while True:
            phi_n, gamma_n = phi + t * step_g, gamma + t * step_v
            e_n = eta(phi_n, gamma_n)
            lp_n = log_post(phi_n, gamma_n, e_n)
            if lp_n >= lp - 1e-9 * abs(lp) or t < 1e-6:
                break
            t /= 2
        moved = t * max(np.abs(step_g).max(), np.abs(step_v).max() if k else 0.0)
        phi, gamma, e, lp = phi_n, gamma_n, e_n, lp_n
        if moved < tol:
            break

    # covariance = inverse of minus the Hessian at the peak, block by block
    _, _, Ainv, AinvB, schur = solve(*blocks(phi, gamma, e))
    S_vv = np.linalg.inv(schur) if k else np.zeros((0, 0))
    S_gv = -AinvB @ S_vv
    S_gg = Ainv + AinvB @ S_vv @ AinvB.transpose(0, 2, 1)
    return Posterior(phi, gamma, S_gg, S_gv, S_vv, it)


# =============================================================================
# 7. The update methods (Q10: recal 3.2, new_only 3.4, joint_v2 3.6)
# =============================================================================

@dataclass
class Visits:
    """What every method needs about a set of visits."""
    g: np.ndarray   # (n,) atlas complaint index
    x: np.ndarray   # (n, 10) atlas design: 1, male, cr(age), male * cr(age)
    s: np.ndarray   # (n,) atlas logit, logit(P_atlas)
    V: np.ndarray   # (n, k) vital features (k = 0 for the no-vitals twins)
    y: np.ndarray   # (n,) died within 30 days

    def __getitem__(self, m) -> "Visits":
        return Visits(self.g[m], self.x[m], self.s[m], self.V[m], self.y[m])

    def without_vitals(self) -> "Visits":
        return Visits(self.g, self.x, self.s, self.V[:, :0], self.y)


def _gamma_prior(k):
    return np.zeros(k), np.eye(k)  # vitals ~ N(0, 1), as in the demo


def fit_recal(d: Visits) -> Posterior:
    """3.2: logit P = a + b * logit(P_atlas) + V . gamma
     * one group for everyone, Xg = (1, atlas logit); a ~ N(0, 1), b ~ N(1, 1), gamma ~ N(0, 1)
     * the prior is centred on the atlas unchanged (a = 0, b = 1, gamma = 0)
    """
    n = len(d.y)
    return fit_laplace(np.zeros(n, int), np.column_stack([np.ones(n), d.s]), d.V, d.y,
                       np.array([[0.0, 1.0]]), np.eye(2)[None], *_gamma_prior(d.V.shape[1]))


def predict_recal(post: Posterior, d: Visits) -> np.ndarray:
    n = len(d.y)
    return post.predict(np.zeros(n, int), np.column_stack([np.ones(n), d.s]), d.V)


def fit_new_only(d: Visits, n_groups: int) -> Posterior:
    """3.4: the atlas's design fitted from scratch on the new visits, ignoring the atlas
     * phi_c ~ N(0, 1) per coefficient, except the intercept ~ N(logit(death rate), 2^2): N(0, 1)
       would pull a ~2% mortality towards 50% (as in 07)
     * a complaint with few visits stays close to this vague prior: the cost of ignoring the atlas
    """
    mg = np.zeros((n_groups, 10))
    mg[:, 0] = logit(d.y.mean())
    Pg = np.tile(np.diag(1 / np.r_[2.0, np.ones(9)] ** 2), (n_groups, 1, 1))
    return fit_laplace(d.g, d.x, d.V, d.y, mg, Pg, *_gamma_prior(d.V.shape[1]))


def fit_joint(d: Visits, atlas: Atlas, tau: float) -> Posterior:
    """3.6 with the atlas posterior in place of the old visits (Q15 option B): a commensurate prior
     * phi_c ~ N(atlas mean_c, atlas cov_c + tau^2 I), gamma ~ N(0, 1)
     * tau = how far this data's coefficients may drift from the atlas's; tau = 0 is 3.3 (atlas
       posterior as prior, unchanged)
     * tau is the same for all 10 coefficients, which are on the spline's raw scale
    """
    Pg = np.linalg.inv(atlas.cov + tau ** 2 * np.eye(10))
    return fit_laplace(d.g, d.x, d.V, d.y, atlas.mean, Pg, *_gamma_prior(d.V.shape[1]))


def predict_grouped(post: Posterior, d: Visits) -> np.ndarray:
    return post.predict(d.g, d.x, d.V)


def log_loss(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pick_tau(train: Visits, val: Visits, atlas: Atlas, grid) -> tuple[float, pd.DataFrame]:
    """Q18 (a): fit joint_v2 on train for each tau in the grid, keep the tau with the lowest
    validation log-loss. The test set is never used to choose."""
    rows = [{"tau": t, "val_log_loss": log_loss(val.y, predict_grouped(fit_joint(train, atlas, t), val))} for t in grid]
    table = pd.DataFrame(rows)
    return float(table.loc[table["val_log_loss"].idxmin(), "tau"]), table


TAU_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, default=OUT_ROOT / "built_subset", help="output folder of 02_build.py")
    ap.add_argument("--out", type=Path, help="default: $SWED_OUT/update_atlas_<built folder name>")
    ap.add_argument("--cutoff", type=float, default=60, help="minutes after arrival a vital may be measured (default 60)")
    ap.add_argument("--years-from", type=int, help="keep only visits from this year on (atlas-overlap check)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = args.out or OUT_ROOT / f"update_atlas_{args.built.name}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"built: {args.built}  out: {out}  cutoff: {args.cutoff:g} min  years from: {args.years_from}")

    counts: dict = {}
    ed = load_visits(args.built, args.years_from, counts)
    value, minutes = first_vitals(ed, args.built)
    coverage_curve(ed, minutes).to_csv(out / "coverage_curve.csv", index=False)

    late = minutes > args.cutoff
    counts[f"first values dropped: measured more than {args.cutoff:g} min after arrival"] = int(late.sum().sum())
    value = value.mask(late)  # a vital first measured after the cutoff counts as missing
    # (the first value is used, so a later value within the cutoff can't exist when the first is late)

    ed["role"] = split_patients(ed, args.seed)
    update = ed["role"].isin(["train", "val"]).to_numpy()
    scale = vital_scale(value, update)
    X = vital_features(value, scale)

    # --- cohort per role ---
    roles = ed.groupby("role").agg(visits=("dead", "size"), patients=("lopnr_swed", "nunique"), deaths=("dead", "sum"))
    roles = roles.reindex(["train", "val", "test"])
    for r in roles.index:
        if roles.loc[r, "deaths"] < MIN_CELL:
            raise SystemExit(f"{r}: fewer than {MIN_CELL} deaths within 30 days, too few to fit or export")
    cohort = roles.apply(lambda s: s.map(round5)).reset_index()
    cohort["death_rate"] = (roles["deaths"] / roles["visits"]).round(4).to_numpy()
    cohort.to_csv(out / "cohort.csv", index=False)

    # --- vitals in the update set ---
    rls = value.loc[update, "rls"].round()
    summary = scale.round(3).reset_index(names="vital")
    summary = pd.concat([summary, pd.DataFrame([{"vital": "rls", "share_present": round(float(rls.notna().mean()), 3)}] + [
        {"vital": col, "share_present": round(float(rls.between(lo, hi).mean()), 4)} for col, (lo, hi) in RLS_LEVELS.items()])])
    summary.to_csv(out / "vitals_summary.csv", index=False)
    pd.DataFrame([{"item": k, "value": round5(v)} for k, v in counts.items()]).to_csv(out / "counts.csv", index=False)

    log("cohort:\n" + cohort.to_string(index=False))
    log("vitals in the update set:\n" + summary.to_string(index=False))
    log(f"{X.shape[1]} vital features: {', '.join(X.columns)}")
    log(f"done. Read every CSV in {out} before export.")


if __name__ == "__main__":
    main()
