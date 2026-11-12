#!/usr/bin/env python3
"""End-to-end evaluation runner for the modular-updating experiments.

    python -m evaluation.orchestrator --built data_pipeline/synthetic/built_synth_subset_50k \
        --out evaluation/results/run_50k --n-boot 1000 --n-boot-subgroup 200

Writes metrics, missingness scoreboards, kappa sensitivity, figures and a
reproducibility manifest under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.cohort import (
    MIN_CELL,
    SEED,
    Cohort,
    Split,
    assert_disjoint,
    complete_case_cohort,
    load_cohort,
    round5,
    save_split_manifest,
    subset_split,
    summarise,
)
from evaluation.metrics import bootstrap_ci, bootstrap_diff
from evaluation.missingness import (
    coobservation_matrix,
    fit_mask_model,
    fit_observation_model,
    hospital_vital_diagnostics,
    imputed_cohorts,
    kappa_sweep,
)
from evaluation.models import LaplaceModel, fit_all, fit_m_ref

MODEL_ORDER = ["M_ref", "M_base", "A", "B", "C", "M_direct"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--built", type=Path, default=Path("data_pipeline/synthetic/built_synth_subset_50k"))
    p.add_argument("--out", type=Path, default=Path("evaluation/results/run"))
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--missing-methods", default="median,knn,mice")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-boot-subgroup", type=int, default=200)
    p.add_argument("--n-boot-kappa", type=int, default=0)
    p.add_argument("--mice-m", type=int, default=5)
    p.add_argument("--kappa-grid", default="auto")
    p.add_argument("--kappa-max", type=float, default=2.5)
    p.add_argument("--kappa-step", type=float, default=0.5)
    p.add_argument("--min-events", type=int, default=10)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    if args.quick:
        args.n_boot = min(args.n_boot, 50)
        args.n_boot_subgroup = min(args.n_boot_subgroup, 20)
        args.mice_m = min(args.mice_m, 2)
        args.no_figures = True
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.missing_methods = [m.strip() for m in args.missing_methods.split(",") if m.strip()]
    if args.kappa_grid == "auto":
        n = int(round(2 * args.kappa_max / args.kappa_step)) + 1
        args.kappas = np.linspace(-args.kappa_max, args.kappa_max, n)
    else:
        args.kappas = np.array([float(x) for x in args.kappa_grid.split(",")])
    return args


def _frame(
    results: dict[str, dict[str, float]],
    model: str,
    n: int,
    events: int,
) -> list[dict]:
    rows = []
    for metric, vals in results.items():
        rows.append(
            {
                "model": model,
                "metric": metric,
                "value": vals["value"],
                "ci_low": vals["lo"],
                "ci_high": vals["hi"],
                "n": round5(n),
                "events": round5(events) if events >= MIN_CELL else None,
            }
        )
    return rows


def _evaluate(
    split: Split,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray | None,
    n_boot: int,
    min_events: int,
    seed: int,
) -> pd.DataFrame:
    patients = split.visits["lopnr_swed"].to_numpy()
    rows = []
    for name, p in predictions.items():
        res = bootstrap_ci(
            split.y,
            p,
            patients,
            groups=groups,
            B=n_boot,
            seed=seed,
            min_events=min_events,
        )
        rows.extend(_frame(res, name, len(split.y), int(split.y.sum())))
    return pd.DataFrame(rows)


def _fit_predict(cohort: Cohort, models: list[str]) -> dict[str, np.ndarray]:
    fitted = fit_all(cohort)
    return {name: fitted[name].predict(cohort["test"]) for name in models}


def _average_predictions(per_imputation: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for name in per_imputation[0]:
        out[name] = np.mean([pred[name] for pred in per_imputation], axis=0)
    return out


def _verdict(
    split: Split,
    predictions: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> str:
    from evaluation.metrics import log_loss

    patients = split.visits["lopnr_swed"].to_numpy()
    losses = {name: log_loss(split.y, p) for name, p in predictions.items()}
    ranked = sorted(losses, key=losses.get)
    best = ranked[0]
    lines = [
        "VERDICT (lower log-loss is better; paired patient bootstrap, 95% CI)",
        f"test visits: {round5(len(split.y))}, deaths: {round5(int(split.y.sum()))}",
        "",
    ]
    for name in ranked:
        if name == best:
            lines.append(f"{name:9s} log-loss {losses[name]:.5f}  (best)")
            continue
        d = bootstrap_diff(
            split.y,
            predictions[best],
            predictions[name],
            patients,
            metric="log_loss",
            B=n_boot,
            seed=seed,
        )
        if d["lo"] <= 0 <= d["hi"]:
            rel = "statistically tied with best"
        elif d["diff"] < 0:
            rel = "best is better"
        else:
            rel = "model is better"
        lines.append(
            f"{name:9s} log-loss {losses[name]:.5f}  diff vs best "
            f"{d['diff']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}]  {rel}"
        )
    return "\n".join(lines) + "\n"


def _write_manifest(out: Path, built: Path, args: argparse.Namespace) -> None:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size})
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "built": str(built),
        "python": sys.version,
        "platform": platform.platform(),
        "args": {k: v for k, v in vars(args).items() if k != "kappas"},
        "kappa_grid": args.kappas.tolist(),
        "files": files,
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        payload["git_commit"] = None
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))


def _write_summary(
    out: Path,
    cohort: Cohort,
    overall: pd.DataFrame,
    missing: pd.DataFrame,
    verdict: str,
    args: argparse.Namespace,
) -> None:
    pivot = (
        overall[overall["metric"].isin(["micro_auc", "log_loss", "bss", "cal_slope", "oe_ratio"])]
        .pivot_table(index="model", columns="metric", values="value")
        .reindex(MODEL_ORDER)
    )
    lines = [
        "# Evaluation run summary",
        "",
        f"- built cohort: `{args.built}`",
        f"- seed: {args.seed}; bootstrap overall {args.n_boot}, subgroups {args.n_boot_subgroup}",
        f"- test visits: {round5(len(cohort['test'].y))}; deaths: {round5(int(cohort['test'].y.sum()))}",
        f"- primary missingness handling: median/mean imputation + observation mask",
        "",
        "## Overall test metrics",
        "",
        pivot.round(5).to_markdown(),
        "",
        "## Verdict",
        "",
        "```",
        verdict.strip(),
        "```",
        "",
        "## Missingness scoreboard (log-loss)",
        "",
    ]
    if not missing.empty:
        mm = missing[missing["metric"] == "log_loss"].pivot_table(
            index="method", columns="model", values="value"
        )
        lines += [mm.round(5).to_markdown(), ""]
    lines += [
        "## Figures",
        "",
        "- `figures/roc_curves.png` — micro discrimination (AUROC per model).",
        "- `figures/precision_recall.png` — precision/recall, more informative for a ~1.5% outcome.",
        "- `figures/calibration.png` — decile calibration with 95% Wilson intervals.",
        "- `figures/forest_vs_M_direct.png` — paired bootstrap differences in log-loss and Brier vs `M_direct`.",
        "- `figures/decision_curve.png` — net benefit across decision thresholds.",
        "- `figures/hospital_auc_heatmap.png` — per-hospital AUROC heterogeneity.",
        "- `figures/missingness_heatmap.png` — `P(vital observed)` by hospital × vital.",
        "- `figures/missingness_scoreboard_heatmap.png` — missingness strategy × model log-loss.",
        "- `figures/kappa_sensitivity.png` — O:E and BSS under the kappa shift (κ<0 shaded).",
        "- `figures/vital_coefficients.png` — fitted vital `(z, z², mask)` coefficients across A/B/C/`M_direct`.",
        "",
        "## Limitations",
        "",
        "- Synthetic SWED data validates code and workflow, not clinical conclusions.",
        "- `kappa < 0` cannot unmask vitals that were never built; the primary sweep is `kappa >= 0`.",
        "- Complete-case uses visits with all five selected vitals observed.",
        "- Complete-case metrics use a different (healthier, complete) test subset, so they are not directly comparable with the full-test scoreboard.",
        "- Primary comparison uses median/mean imputation + observation mask; other strategies are sensitivity analyses.",
        "- `report.tex` still uses the older `M0/M_base/M_modular/M_direct` taxonomy.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    cohort = load_cohort(args.built, seed=args.seed)
    assert_disjoint(cohort)
    summarise(cohort).to_csv(out / "cohort_summary.csv", index=False)
    save_split_manifest(cohort, out / "split_manifest.json", args.built, seed=args.seed)

    fitted = fit_all(cohort)
    predictions = {name: fitted[name].predict(cohort["test"]) for name in args.models}
    test = cohort["test"]
    assert all(len(p) == len(test.y) for p in predictions.values()), "models scored different rows"

    np.savez_compressed(
        out / "test_predictions.npz",
        y=test.y,
        patient_id=test.visits["lopnr_swed"].to_numpy(),
        hospital=test.visits["ed_hospital"].to_numpy().astype(str),
        complaint=test.visits["complaint"].to_numpy().astype(str),
        **{f"p_{name}": p for name, p in predictions.items()},
    )

    groups = test.visits["complaint"].to_numpy()
    overall = _evaluate(test, predictions, groups, args.n_boot, args.min_events, args.seed)
    overall.to_csv(out / "metrics_overall.csv", index=False)

    # Per-hospital
    hosp_rows = []
    for hospital in sorted(test.visits["ed_hospital"].dropna().unique()):
        keep = np.flatnonzero(test.visits["ed_hospital"].to_numpy() == hospital)
        if int(test.y[keep].sum()) < args.min_events:
            continue
        sub = subset_split(test, keep)
        res = _evaluate(
            sub,
            {name: p[keep] for name, p in predictions.items()},
            None,
            args.n_boot_subgroup,
            args.min_events,
            args.seed,
        )
        res.insert(0, "hospital", hospital)
        hosp_rows.append(res)
    hosp = pd.concat(hosp_rows, ignore_index=True) if hosp_rows else pd.DataFrame()
    hosp.to_csv(out / "metrics_by_hospital.csv", index=False)

    # Per-complaint
    comp_rows = []
    for complaint in pd.unique(groups):
        keep = np.flatnonzero(groups == complaint)
        if int(test.y[keep].sum()) < args.min_events:
            continue
        sub = subset_split(test, keep)
        res = _evaluate(
            sub,
            {name: p[keep] for name, p in predictions.items()},
            None,
            args.n_boot_subgroup,
            args.min_events,
            args.seed,
        )
        res.insert(0, "complaint", complaint)
        comp_rows.append(res)
    comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
    comp.to_csv(out / "metrics_by_complaint.csv", index=False)

    # Baseline verification and strategy comparison
    patients = test.visits["lopnr_swed"].to_numpy()
    base_rows = []
    if {"M_ref", "M_base"} <= set(predictions):
        for name in ("M_ref", "M_base"):
            sub = overall[overall["model"] == name]
            for _, r in sub.iterrows():
                base_rows.append({"comparison": "baseline", "model": name, **r.to_dict()})
        for metric in ("log_loss", "brier", "micro_auc"):
            d = bootstrap_diff(
                test.y, predictions["M_base"], predictions["M_ref"], patients,
                metric=metric, B=min(args.n_boot, 500), seed=args.seed,
            )
            base_rows.append(
                {
                    "comparison": "M_base_minus_M_ref",
                    "model": metric,
                    "value": d["diff"],
                    "ci_low": d["lo"],
                    "ci_high": d["hi"],
                }
            )
    pd.DataFrame(base_rows).to_csv(out / "baseline_verification.csv", index=False)

    strat_rows = []
    for name in ("A", "B", "C", "M_direct"):
        if name not in predictions:
            continue
        sub = overall[overall["model"] == name]
        for _, r in sub.iterrows():
            strat_rows.append({"model": name, **r.to_dict()})
        if name != "M_direct" and "M_direct" in predictions:
            for metric in ("log_loss", "brier"):
                d = bootstrap_diff(
                    test.y, predictions[name], predictions["M_direct"], patients,
                    metric=metric, B=min(args.n_boot, 500), seed=args.seed,
                )
                strat_rows.append(
                    {
                        "model": f"{name}_minus_M_direct",
                        "metric": metric,
                        "value": d["diff"],
                        "ci_low": d["lo"],
                        "ci_high": d["hi"],
                    }
                )
    pd.DataFrame(strat_rows).to_csv(out / "strategy_comparison.csv", index=False)

    # Missingness diagnostics
    hospital_vital_diagnostics(cohort).to_csv(out / "missingness_diagnostics.csv", index=False)
    coobservation_matrix(cohort).to_csv(out / "coobservation_matrix.csv")

    # Missingness scoreboard
    board = []
    tab = _evaluate(test, predictions, groups, args.n_boot_subgroup, args.min_events, args.seed)
    tab["method"] = "mean_mask"
    board.append(tab)
    for method in args.missing_methods:
        variants = imputed_cohorts(cohort, method, m=args.mice_m, seed=args.seed)
        preds = [_fit_predict(c, args.models) for c in variants]
        avg = _average_predictions(preds) if len(preds) > 1 else preds[0]
        tab = _evaluate(test, avg, groups, args.n_boot_subgroup, args.min_events, args.seed)
        tab["method"] = f"{method}_mask"
        board.append(tab)
    cc = complete_case_cohort(cohort)
    if int(cc["new"].y.sum()) >= args.min_events:
        cc_pred = _fit_predict(cc, args.models)
        tab = _evaluate(
            cc["test"],
            cc_pred,
            cc["test"].visits["complaint"].to_numpy(),
            args.n_boot_subgroup,
            args.min_events,
            args.seed,
        )
        tab["method"] = "complete_case"
        board.append(tab)
    mask_model = fit_mask_model(cohort, fit_m_ref(cohort))
    mask_pred = {"Mask": mask_model.predict(test)}
    tab = _evaluate(test, mask_pred, groups, args.n_boot_subgroup, args.min_events, args.seed)
    tab["method"] = "mask_interactions"
    board.append(tab)
    missing = pd.concat(board, ignore_index=True)
    missing.to_csv(out / "missingness_scoreboard.csv", index=False)

    # Kappa sensitivity (primary sweep kappa >= 0)
    obs_model, _ = fit_observation_model(cohort)
    obs = obs_model.predict(test)
    print(f"observation model mean P(all vitals)={obs.mean():.3f}", flush=True)
    kappa = kappa_sweep(cohort, predictions, args.kappas, n_boot=args.n_boot_kappa, seed=args.seed)
    kappa["primary"] = kappa["kappa"] >= 0
    kappa.to_csv(out / "kappa_sensitivity.csv", index=False)

    verdict = _verdict(test, predictions, min(args.n_boot, 500), args.seed)
    (out / "verdict.txt").write_text(verdict)

    if not args.no_figures:
        from evaluation.plots import make_all

        make_all(out, cohort=cohort)

    config = {
        "built": str(args.built),
        "seed": args.seed,
        "models": args.models,
        "missing_methods": args.missing_methods,
        "n_boot": args.n_boot,
        "n_boot_subgroup": args.n_boot_subgroup,
        "n_boot_kappa": args.n_boot_kappa,
        "mice_m": args.mice_m,
        "kappa_grid": args.kappas.tolist(),
        "min_events": args.min_events,
        "quick": args.quick,
        "split": "40/40/20 patients, seed {}".format(args.seed),
        "base_feature_names": cohort.base_feature_names,
        "vital_feature_names": cohort.vital_feature_names,
        "scaling": cohort.scaling,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    _write_summary(out, cohort, overall, missing, verdict, args)
    _write_manifest(out, args.built, args)

    print(verdict)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
