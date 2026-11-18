#!/usr/bin/env python3
"""Diagnostic plots for a completed evaluation run.

    python -m evaluation.plots --run evaluation/results/run_50k

Reads ``test_predictions.npz`` and the result CSVs, then writes figures that
describe discrimination (ROC/PR), calibration (with binomial intervals),
paired model differences, hospital/missingness heterogeneity, kappa
sensitivity and the fitted vital coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from evaluation.metrics import bootstrap_diff, calibration_table, log_loss, brier
from evaluation.models import fit_all

MODEL_ORDER = ["M_ref", "M_base", "A", "B", "C", "M_direct"]
COLORS = {
    "M_ref": "#6c757d",
    "M_base": "#adb5bd",
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#d62728",
    "M_direct": "#9467bd",
    "Mask": "#ff7f0e",
}


def _load(run: Path) -> dict:
    z = np.load(run / "test_predictions.npz", allow_pickle=True)
    models = [k[2:] for k in z.files if k.startswith("p_")]
    models = [m for m in MODEL_ORDER if m in models] + [m for m in models if m not in MODEL_ORDER]
    out = {
        "y": z["y"],
        "patient_id": z["patient_id"],
        "hospital": z["hospital"].astype(str),
        "complaint": z["complaint"].astype(str),
        "pred": {m: z[f"p_{m}"] for m in models},
        "models": models,
    }
    for name in (
        "metrics_overall",
        "metrics_by_hospital",
        "metrics_by_complaint",
        "missingness_scoreboard",
        "missingness_diagnostics",
        "kappa_sensitivity",
        "strategy_comparison",
    ):
        path = run / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def _colors(models: list[str]) -> list[str]:
    return [COLORS.get(m, None) for m in models]


def plot_roc(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in data["models"]:
        p = data["pred"][model]
        fpr, tpr, _ = roc_curve(data["y"], p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS.get(model), label=f"{model} (AUC {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    prevalence = float(np.mean(data["y"]))
    for model in data["models"]:
        p = data["pred"][model]
        prec, rec, _ = precision_recall_curve(data["y"], p)
        ap = average_precision_score(data["y"], p)
        ax.plot(rec, prec, color=COLORS.get(model), label=f"{model} (AP {ap:.3f})")
    ax.axhline(prevalence, color="k", linestyle="--", linewidth=1, label=f"prevalence {prevalence:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (positive predictive value)")
    ax.set_title("Precision-recall curves (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def plot_calibration(data: dict, path: Path, bins: int = 10, B: int = 500) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    tops = []
    pid = data.get("patient_id")
    if pid is None:
        pid = np.arange(len(data["y"]))
    for model in data["models"]:
        p = data["pred"][model]
        tab = calibration_table(data["y"], p, pid, bins=bins, B=B, seed=42)
        lo, hi = zip(*[_wilson(int(k), int(n)) for k, n in zip(tab["k"], tab["n"])])
        yerr = np.vstack([tab["obs"] - np.asarray(lo), np.asarray(hi) - tab["obs"]])
        ax.fill_between(tab["pred"], tab["lo"], tab["hi"], color=COLORS.get(model), alpha=0.14, lw=0)
        ax.errorbar(
            tab["pred"],
            tab["obs"],
            yerr=yerr,
            fmt="o-",
            color=COLORS.get(model),
            label=model,
            capsize=2,
            markersize=4,
            linewidth=1,
        )
        tops.append(float(tab["pred"].max()))
    lim = max(0.05, min(0.15, max(tops) * 1.1))
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="ideal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted risk (decile)")
    ax.set_ylabel("Observed death rate (bars: Wilson; band: patient bootstrap)")
    ax.set_title("Calibration (test, deciles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forest(data: dict, path: Path, reference: str = "M_direct", B: int = 500) -> None:
    models = [m for m in data["models"] if m != reference]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(models) + 2.2), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("log_loss", "brier"),
        ("Log-loss difference", "Brier difference"),
    ):
        for i, model in enumerate(models):
            d = bootstrap_diff(
                data["y"],
                data["pred"][reference],
                data["pred"][model],
                data["patient_id"],
                metric=metric,
                B=B,
            )
            ax.errorbar(
                d["diff"],
                i,
                xerr=[[d["diff"] - d["lo"]], [d["hi"] - d["diff"]]],
                fmt="o",
                color=COLORS.get(model),
                capsize=3,
            )
        ax.axvline(0, color="k", linewidth=1)
        ax.set_title(f"{title} vs {reference}")
        ax.set_xlabel("positive = model better (lower loss)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    fig.suptitle(f"Paired patient bootstrap (95% CI) vs {reference}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    fmt: str = "{:.3f}",
    cmap: str = "viridis",
    center: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * matrix.shape[1] + 2), max(3.5, 0.5 * matrix.shape[0] + 1.5)))
    vals = matrix.to_numpy(dtype=float)
    if center is not None:
        finite = vals[np.isfinite(vals)]
        vmax = max(abs(finite).max(), 1e-9) if len(finite) else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if (norm is None and v < np.nanmean(vals) + 0.5 * np.nanstd(vals)) else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hospital_heatmap(data: dict, path: Path, metric: str = "micro_auc") -> None:
    df = data["metrics_by_hospital"]
    if df.empty:
        return
    m = df[df["metric"] == metric].pivot(index="model", columns="hospital", values="value")
    m = m.reindex([x for x in data["models"] if x in m.index])
    _heatmap(m, path, f"{metric} by hospital (test)", fmt="{:.3f}", cmap="magma")


def plot_missingness_heatmap(data: dict, path: Path) -> None:
    df = data["missingness_diagnostics"]
    if df.empty:
        return
    sub = df[df["split"] == "test"]
    if sub.empty:
        sub = df
    m = sub.pivot_table(index="hospital", columns="vital", values="p_observed")
    _heatmap(m, path, "P(vital observed) by hospital (test)", fmt="{:.2f}", cmap="viridis")


def plot_scoreboard_heatmap(data: dict, path: Path) -> None:
    df = data["missingness_scoreboard"]
    if df.empty:
        return
    m = df[df["metric"] == "log_loss"].pivot_table(index="method", columns="model", values="value")
    _heatmap(m, path, "Missingness strategies: log-loss (lower is better)", fmt="{:.4f}", cmap="magma_r")


def plot_decision_curve(data: dict, path: Path) -> None:
    thresholds = np.linspace(0.001, 0.10, 60)
    n = len(data["y"])
    prev = float(np.mean(data["y"]))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in data["models"]:
        p = data["pred"][model]
        nb = []
        for t in thresholds:
            pos = p >= t
            tp = float(np.sum(pos & (data["y"] == 1)))
            fp = float(np.sum(pos & (data["y"] == 0)))
            nb.append(tp / n - fp / n * (t / (1 - t)))
        ax.plot(thresholds, nb, color=COLORS.get(model), label=model)
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all, "k--", linewidth=1, label="treat all")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kappa(data: dict, path: Path) -> None:
    df = data["kappa_sensitivity"]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ("oe_ratio", "bss"), ("O:E ratio", "Brier skill score")):
        for model in data["models"]:
            sub = df[df["model"] == model].sort_values("kappa")
            if sub.empty:
                continue
            ax.plot(sub["kappa"], sub[col], marker="o", markersize=3,
                    color=COLORS.get(model), label=model)
        ax.axvline(0, color="k", linewidth=1)
        ax.axvspan(df["kappa"].min() - 0.1, 0, color="grey", alpha=0.12)
        ax.axhline(1, color="k", linestyle=":", linewidth=0.8)
        ax.set_xlabel("kappa shift applied to logit(p)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Kappa sensitivity (shaded: kappa < 0, not primary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _vital_block(model, cohort) -> np.ndarray | None:
    k_vit = len(cohort.vital_feature_names) + len(cohort.vital_names)
    if model.name in ("A",):
        return model.theta[:k_vit]
    if model.name in ("B",):
        return model.theta[2:]
    if model.name in ("C", "M_direct"):
        k_base = len(cohort.base_feature_names) + 1
        return model.theta[k_base:]
    return None


def plot_vital_coefficients(cohort, path: Path) -> None:
    fitted = fit_all(cohort)
    names = cohort.vital_feature_names + [f"{v}_observed" for v in cohort.vital_names]
    rows, labels = [], []
    for name in ("A", "B", "C", "M_direct"):
        block = _vital_block(fitted[name], cohort)
        if block is None:
            continue
        rows.append(block)
        labels.append(name)
    if not rows:
        return
    m = pd.DataFrame(np.vstack(rows), index=labels, columns=names)
    _heatmap(m, path, "Fitted vital coefficients (z, z², observation mask)", fmt="{:.2f}",
             cmap="coolwarm", center=0.0)


def make_all(run: str | Path, cohort=None) -> None:
    run = Path(run)
    figdir = run / "figures"
    figdir.mkdir(exist_ok=True)
    data = _load(run)
    plot_roc(data, figdir / "roc_curves.png")
    plot_pr(data, figdir / "precision_recall.png")
    plot_calibration(data, figdir / "calibration.png")
    plot_forest(data, figdir / "forest_vs_M_direct.png")
    plot_decision_curve(data, figdir / "decision_curve.png")
    plot_hospital_heatmap(data, figdir / "hospital_auc_heatmap.png")
    plot_missingness_heatmap(data, figdir / "missingness_heatmap.png")
    plot_scoreboard_heatmap(data, figdir / "missingness_scoreboard_heatmap.png")
    plot_kappa(data, figdir / "kappa_sensitivity.png")
    if cohort is None:
        cfg = json.loads((run / "run_config.json").read_text())
        from evaluation.cohort import load_cohort

        cohort = load_cohort(cfg["built"], seed=cfg["seed"])
    plot_vital_coefficients(cohort, figdir / "vital_coefficients.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("evaluation/results/run_50k"))
    p.add_argument("--built", type=Path, default=None, help="override built cohort path")
    args = p.parse_args()
    cohort = None
    if args.built is not None:
        from evaluation.cohort import load_cohort

        cohort = load_cohort(args.built)
    make_all(args.run, cohort=cohort)
    print(f"wrote figures to {args.run / 'figures'}")


if __name__ == "__main__":
    main()
