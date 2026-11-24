from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_curve


def plot_path(out_path, suffix):
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}{suffix}.png")


def plot_calibration(y, probabilities, slope, intercept, out_path, title):
    bins = min(10, len(y))
    ranked_bins = pd.qcut(
        pd.Series(probabilities).rank(method="first"),
        q=bins,
        labels=False,
        duplicates="drop",
    )
    grouped = pd.DataFrame(
        {"bin": ranked_bins, "observed": y, "predicted": probabilities}
    ).groupby("bin", observed=True).mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.plot(
        grouped["predicted"],
        grouped["observed"],
        "o-",
        color="#2463a6",
        label="Observed by risk decile",
    )
    if np.isfinite(slope) and np.isfinite(intercept):
        risk = np.linspace(
            max(probabilities.min(), 1e-8),
            min(probabilities.max(), 1 - 1e-8),
            200,
        )
        logits = np.log(risk / (1 - risk))
        ax.plot(
            risk,
            expit(intercept + slope * logits),
            color="#d45a31",
            label=f"Calibration fit (slope={slope:.2f})",
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event rate",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _roc_band(labels, probabilities, average):
    grid = np.linspace(0, 1, 101)
    curves = []
    for draw in probabilities:
        if average == "micro":
            groups = [(labels.ravel(), draw.ravel())]
        elif average == "macro":
            groups = [
                (labels[:, index], draw[:, index])
                for index in range(labels.shape[1])
                if 0 < labels[:, index].sum() < len(labels)
            ]
        else:
            groups = [(labels, draw)]

        draw_curves = []
        for y, score in groups:
            if np.unique(y).size != 2:
                continue
            fpr, tpr, _ = roc_curve(y, score)
            curve = np.interp(grid, fpr, tpr)
            curve[0], curve[-1] = 0, 1
            draw_curves.append(curve)
        if draw_curves:
            curves.append(np.mean(draw_curves, axis=0))

    if not curves:
        return grid, None, None, None
    curves = np.asarray(curves)
    return (
        grid,
        curves.mean(axis=0),
        np.quantile(curves, 0.025, axis=0),
        np.quantile(curves, 0.975, axis=0),
    )


def plot_roc(labels, probabilities, metrics, target_names, out_path):
    multi_label = labels.shape[1] > 1
    fig, ax = plt.subplots(figsize=(7, 6))
    curves = ("micro", "macro") if multi_label else (target_names[0],)
    for name in curves:
        average = name if multi_label else "binary"
        y = labels if multi_label else labels[:, 0]
        scores = probabilities if multi_label else probabilities[:, :, 0]
        grid, mean, lower, upper = _roc_band(y, scores, average)
        if mean is None:
            continue
        row = metrics.loc[metrics["target"] == name].iloc[0]
        ax.plot(
            grid,
            mean,
            label=(
                f"{name.title()} AUC {row['roc_auc']:.3f} "
                f"(95% CrI {row['roc_auc_lower_95']:.3f}–"
                f"{row['roc_auc_upper_95']:.3f})"
            ),
        )
        ax.fill_between(grid, lower, upper, alpha=0.18)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="No discrimination")
    if len(ax.lines) == 1:
        ax.text(
            0.5,
            0.5,
            "ROC-AUC is undefined: test labels contain only one class.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Diagnosis micro/macro ROC" if multi_label else f"{target_names[0]} ROC",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
