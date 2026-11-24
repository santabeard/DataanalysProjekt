# Path: BLR vitals model — divide-and-conquer implementation breakdown

> Derived from `DESIGN.md`'s settled Plan (Q1–Q10, build order). This is a structural breakdown
> for Top Baldy to implement independently — no code here, just what the subproblems are, how
> they split, and how they combine back together.

## Why this problem divides cleanly

The three models (mortality, diagnosis tree level 1, diagnosis tree levels 2/3) all read the same
prepared data and use the same shared module, but fitting one doesn't depend on the result of
fitting another — that's the independent-subproblems property divide-and-conquer needs. Everything
upstream of those three fits (data prep, environment checks, the shared module) is a one-time
**Divide** cost paid once; the three fits are the **Conquer** step, done separately; exporting
comparable summaries at the end is the **Combine** step.

## Dependency map

```
Phase 0 (prerequisites — independent of each other, do both before anything else)
  P0.a  Verify scipy is importable on Bianca's python_ML_packages/3.9.5-cpu
  P0.b  Add ed_hospital to 04_train_clean.py's keep_cols; re-run on Bianca; confirm
        covered-site row count (~989K estimate) against the real column

Phase 1 (shared module — itself divides into 4 independent pieces, then combines into one module)
  P1.a  Formula builder — two variants: full interaction (mortality, tree level 1)
        and additive (tree levels 2/3)
  P1.b  Weighted-likelihood mechanism, using complaint_weight (per ed_atlas_impl's
        mechanism, applied to this project's own weight column)
  P1.c  Age-spline knot constants [22, 43, 65, 88] (per ed_atlas_impl, reused as-is)
  P1.d  Standardization fit/apply pair (z-score, fit on the Stage-05 training split)
  → Combine: assemble P1.a–P1.d into the one shared module (per DESIGN.md Q8's
    codebase-organization decision). Base case check: run a synthetic smoke test
    on the assembled module BEFORE touching Bianca — this is the cheapest point
    to catch a broken formula/likelihood/scaling interaction.

Phase 2 (conditional — only needed if NUTS proves infeasible and the Laplace
         fallback from Q7 is actually invoked; depends on P0.a's result)
  P2    Fix the Laplace/hierarchical-prior degeneracy: the joint-mode Laplace
        approximation is degenerate under Q5's hierarchical prior as the
        hierarchical scale σ→0. Decide fix-σ (empirical Bayes / type-II ML, or a
        short NUTS run on a subsample) vs. integrate σ out, then implement inside
        the shared module's Laplace path. Test on synthetic data before Bianca.

Phase 3 (data gap-fill — depends on P0.b; needed before any vitals-using fit)
  P3    Decide and implement the partial within-site vitals gap fill: reuse the
        missing-indicator + training-split-mean approach from Q10, applied only
        within the covered-site rows. Verify-first: read 02_vitals_join.py's
        Bianca log for the real per-vital coverage % at the 3 covered hospitals.

Phase 4 (the three independent model fits — the actual Conquer step; each depends
         on Phase 1's shared module and Phase 3's gap-filled data, and on Phase 2
         only if the Laplace path is used)
  P4.a  Mortality model (binary, died_30d) — full interaction formula, all
        ~1.8M train_clean rows for non-vitals terms, vitals terms on the
        ~989K covered-site subset. Fit this one first: smallest and most
        informative, and it validates the whole pipeline end-to-end before
        spending Bianca time on the other two.
  P4.b  Diagnosis tree level 1 (binary, NODIAG vs. any) — same formula shape
        and row/vitals split as mortality. Independent of P4.a and P4.c; can
        start as soon as P4.a has validated the pipeline, without waiting for
        P4.a's actual fit to finish if Top Baldy has the bandwidth to run both.
  P4.c  Diagnosis tree levels 2/3 (chapter, then exact top-K code within
        chapter) — additive formula per Q9. Independent of P4.a and P4.b in
        the same way.

Phase 5 (Combine step)
  P5    Export posterior/MAP summaries from all three fits into one
        ed_atlas-comparable format. Decide convergence/output criteria here
        (ed_atlas's own mortality fit uses 1 chain, so no R-hat — a light
        standard is consistent with that precedent, not a gap to over-solve).
```

## Recommended execution order

1. **P0.a and P0.b** — run in either order, or in parallel; they don't depend on each other.
2. **P1.a through P1.d, then the Phase 1 combine** (shared module + synthetic smoke test). Nothing
   below this line should start before the smoke test passes — it's the cheap, local base case
   that catches mistakes before they cost real Bianca time.
3. **P3** — gated only on P0.b (needs `ed_hospital` to exist in the data).
4. **P2** — only if P0.a's scipy check is being relied on because gather-NUTS (Q1) turns out
   infeasible on Bianca. Skip entirely if NUTS works.
5. **P4.a (mortality)** — first real Bianca fit. Treat its timing as the first real (non-laptop)
   cost measurement for the whole project.
6. **P4.b and P4.c** — once P4.a has validated the pipeline, these can run in any order, or
   concurrently if there's bandwidth for it, since they don't share state with each other or with
   P4.a beyond already-built Phase 1/3 artifacts.
7. **P5** — only after all three of P4.a/b/c have usable fitted output.

## Watch-fors specific to this plan

- **Don't skip the Phase 1 smoke test.** It's the one point in this plan equivalent to "test the
  base case and one level of recursion before trusting the full recursion" — the cheapest place to
  catch a formula/likelihood/scaling bug, before any Bianca time is spent on it.
- **Combine-step scope creep (P5).** The export/summary format could expand into its own project if
  convergence criteria become contentious. Keep it lightweight and consistent with ed_atlas's own
  precedent (1 chain, no R-hat) rather than over-engineering a comparison standard.
- **Check each Phase 4 fit independently before moving to the next**, not just at the end — sign and
  shape sanity against `ed_atlas`'s published shapes (where comparable) catches a broken shared
  module once, rather than discovering the same bug three separate times.
- **Age-spline extrapolation** outside the `[22, 88]` knot range: decide clip-vs-accept before
  Phase 4, not during or after — this is a cheap decision now and an awkward one to revisit once
  three fits already depend on whichever behavior ships.

## Open build-time decisions carried over from DESIGN.md

These aren't blocking the order above, but each is a concrete thing to resolve at the point in the
plan where it first matters, not deferred indefinitely:

- scipy availability on Bianca — P0.a, blocks the entire Laplace fallback path if missing.
- Laplace/hierarchical-prior fix choice — P2, only if reached.
- Real per-vital coverage % at the 3 covered hospitals — P3, needed before the gap-fill is coded.
- Bianca job resource limits (cores/RAM/walltime) — unmeasured; P4.a's fit is the first real
  measurement.
- Coding/identifiability scheme (patsy default vs. explicit) — low-stakes, revisit only if it
  actually causes a fitting problem.
- Plausibility-clipping bounds for vitals — pick clinically sensible bounds during P3, note them
  wherever the gap-fill is implemented.
