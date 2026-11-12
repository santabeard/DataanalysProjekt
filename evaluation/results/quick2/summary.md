# Evaluation run summary

- built cohort: `data_pipeline/synthetic/built_synth_subset_50k`
- seed: 42; bootstrap overall 50, subgroups 20
- test visits: 35660; deaths: 545
- primary missingness handling: median/mean imputation + observation mask

## Overall test metrics

| model    |     bss |   cal_slope |   log_loss |   micro_auc |   oe_ratio |
|:---------|--------:|------------:|-----------:|------------:|-----------:|
| M_ref    | 0.02281 |     1.09703 |    0.07022 |     0.79327 |    1.0236  |
| M_base   | 0.02331 |     1.01181 |    0.07014 |     0.79397 |    1.04307 |
| A        | 0.05689 |     0.96101 |    0.06619 |     0.8301  |    1.03864 |
| B        | 0.05633 |     0.91301 |    0.06619 |     0.83097 |    1.08599 |
| C        | 0.05656 |     0.93878 |    0.06613 |     0.83121 |    1.05603 |
| M_direct | 0.05586 |     0.91309 |    0.06616 |     0.83168 |    1.07815 |

## Verdict

```
VERDICT (lower log-loss is better; paired patient bootstrap, 95% CI)
test visits: 35660, deaths: 545

C         log-loss 0.06613  (best)
M_direct  log-loss 0.06616  diff vs best -0.00003 [-0.00010, +0.00005]  statistically tied with best
A         log-loss 0.06619  diff vs best -0.00005 [-0.00012, +0.00002]  statistically tied with best
B         log-loss 0.06619  diff vs best -0.00006 [-0.00016, +0.00000]  statistically tied with best
M_base    log-loss 0.07014  diff vs best -0.00400 [-0.00480, -0.00261]  best is better
M_ref     log-loss 0.07022  diff vs best -0.00409 [-0.00485, -0.00271]  best is better
```

## Missingness scoreboard (log-loss)

| method            |         A |         B |         C |    M_base |   M_direct |     M_ref |      Mask |
|:------------------|----------:|----------:|----------:|----------:|-----------:|----------:|----------:|
| complete_case     |   0.04422 |   0.04431 |   0.0444  |   0.06611 |    0.04507 |   0.06621 | nan       |
| knn_mask          |   0.0664  |   0.06651 |   0.06637 |   0.07014 |    0.06647 |   0.07022 | nan       |
| mask_interactions | nan       | nan       | nan       | nan       |  nan       | nan       |   0.06621 |
| mean_mask         |   0.06619 |   0.06619 |   0.06613 |   0.07014 |    0.06616 |   0.07022 | nan       |
| median_mask       |   0.06603 |   0.0662  |   0.066   |   0.07014 |    0.06615 |   0.07022 | nan       |
| mice_mask         |   0.06618 |   0.06619 |   0.06613 |   0.07014 |    0.06616 |   0.07022 | nan       |

## Limitations

- Synthetic SWED data validates code and workflow, not clinical conclusions.
- `kappa < 0` cannot unmask vitals that were never built; the primary sweep is `kappa >= 0`.
- Complete-case uses visits with all five selected vitals observed.
- Primary comparison uses median/mean imputation + observation mask; other strategies are sensitivity analyses.
- `report.tex` still uses the older `M0/M_base/M_modular/M_direct` taxonomy.
