#!/usr/bin/env python3
"""Core discrimination, calibration and scoring metrics with bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(y, p))
    r = pd.Series(p).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auc(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    min_events: int = 10,
) -> tuple[float, int]:
    """Unweighted mean AUC over groups with enough events in both classes."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    scores = []
    for g in pd.unique(groups):
        if pd.isna(g):
            continue
        m = groups == g
        yy = y[m]
        n1, n0 = int(yy.sum()), int(len(yy) - yy.sum())
        if n1 < min_events or n0 < min_events:
            continue
        scores.append(auc_score(yy, np.asarray(p)[m]))
    scores = [s for s in scores if np.isfinite(s)]
    return (float(np.mean(scores)) if scores else float("nan"), len(scores))


def oe_ratio(y: np.ndarray, p: np.ndarray) -> float:
    expected = float(np.sum(p))
    return float(np.sum(y) / expected) if expected > 0 else float("nan")


def _fit_calibration(y: np.ndarray, lp: np.ndarray) -> tuple[float, float]:
    """Newton fit of ``y ~ sigmoid(alpha + beta * logit(p))``."""
    X = np.column_stack([np.ones(len(y)), lp])
    beta = np.array([0.0, 1.0])
    ridge = 1e-8 * np.eye(2)
    for _ in range(100):
        eta = np.clip(X @ beta, -30.0, 30.0)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pr * (1 - pr), 1e-10, None)
        grad = X.T @ (y - pr) - ridge @ beta
        H = X.T @ (X * w[:, None]) + ridge
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e6:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return _fit_calibration(np.asarray(y, dtype=float), _logit(p))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    base = float(np.mean(y) * (1 - np.mean(y)))
    return float(1 - brier(y, p) / base) if base > 0 else float("nan")


def nagelkerke(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    n = len(y)
    ll1 = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ybar = float(np.mean(y))
    if ybar in (0.0, 1.0):
        return float("nan")
    ll0 = float(n * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar)))
    denom = 1 - np.exp(2 * ll0 / n)
    if denom == 0:
        return float("nan")
    return float((1 - np.exp(2 * (ll0 - ll1) / n)) / denom)


METRICS = [
    "micro_auc",
    "macro_auc",
    "oe_ratio",
    "cal_intercept",
    "cal_slope",
    "log_loss",
    "brier",
    "bss",
    "nagelkerke",
]


def all_metrics(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray | None = None,
    min_events: int = 10,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    alpha, beta = calibration_intercept_slope(y, p)
    out = {
        "micro_auc": auc_score(y, p),
        "macro_auc": macro_auc(y, p, groups, min_events)[0] if groups is not None else float("nan"),
        "oe_ratio": oe_ratio(y, p),
        "cal_intercept": alpha,
        "cal_slope": beta,
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "bss": brier_skill(y, p),
        "nagelkerke": nagelkerke(y, p),
    }
    return out


def _patient_blocks(patient_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    codes, uniques = pd.factorize(patient_ids, sort=False)
    blocks = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    return uniques, blocks


def _stratified_rows(
    uniques: np.ndarray,
    blocks: list[np.ndarray],
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    block_outcome = np.array([y[b].max() for b in blocks])
    parts = []
    for stratum in (0, 1):
        members = np.flatnonzero(block_outcome == stratum)
        if len(members) == 0:
            continue
        draw = rng.choice(members, size=len(members), replace=True)
        parts.extend(blocks[i] for i in draw)
    return np.concatenate(parts) if parts else np.arange(0)


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    groups: np.ndarray | None = None,
    B: int = 1000,
    seed: int = 42,
    min_events: int = 10,
) -> dict[str, dict[str, float]]:
    """Point estimates plus 95% patient-level, outcome-stratified bootstrap CIs."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups) if groups is not None else None
    point = all_metrics(y, p, groups, min_events)

    uniques, blocks = _patient_blocks(np.asarray(patient_ids))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(B):
        rows = _stratified_rows(uniques, blocks, y, rng)
        if len(rows) == 0 or y[rows].min() == y[rows].max():
            continue
        m = all_metrics(y[rows], p[rows], groups[rows] if groups is not None else None, min_events)
        for k in METRICS:
            draws[k].append(m[k])

    out = {}
    for k in METRICS:
        arr = np.asarray(draws[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            lo = hi = float("nan")
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {"value": point[k], "lo": float(lo), "hi": float(hi)}
    return out


def bootstrap_diff(
    y: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    patient_ids: np.ndarray,
    metric: str = "log_loss",
    B: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Paired patient bootstrap of ``metric(a) - metric(b)``."""
    y = np.asarray(y, dtype=float)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    fn = {
        "log_loss": log_loss,
        "brier": brier,
        "micro_auc": auc_score,
        "oe_ratio": oe_ratio,
    }[metric]
    point = float(fn(y, pa) - fn(y, pb))
    uniques, blocks = _patient_blocks(np.asarray(patient_ids))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        rows = _stratified_rows(uniques, blocks, y, rng)
        if len(rows) == 0 or y[rows].min() == y[rows].max():
            continue
        diffs.append(float(fn(y[rows], pa[rows]) - fn(y[rows], pb[rows])))
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (float("nan"), float("nan"))
    return {
        "diff": point,
        "lo": float(lo),
        "hi": float(hi),
        "p_a_better": float(np.mean(arr < 0)) if len(arr) else float("nan"),
    }


def metrics_frame(
    results: dict[str, dict[str, float]],
    name: str,
    n: int,
    events: int,
) -> pd.DataFrame:
    rows = []
    for metric, vals in results.items():
        rows.append(
            {
                "model": name,
                "metric": metric,
                "value": vals["value"],
                "ci_low": vals["lo"],
                "ci_high": vals["hi"],
                "n": n,
                "events": events,
            }
        )
    return pd.DataFrame(rows)
