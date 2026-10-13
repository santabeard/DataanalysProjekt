"""ED Atlas: Bayesian logistic regression priors for ED triage.

Provides pre-computed predictive distributions for patients presenting
to the emergency department, conditioned on age, sex, and presenting
complaint (SNOMED CT concept ID).

Two outcome types:
  - Diagnosis (ICD-10 codes within 30 days): from atlas_logitnormals.parquet
  - Death (30-day mortality): from atlas_mortality_logitnormals.parquet

Each cell stores a LogitNormal(mu, sigma) predictive distribution.

Three levels of access:
  1. predict() / predict_death()
     Posterior predictive mean: sigmoid(mu). Fast table lookup.

  2. predict_distribution() / predict_death_distribution()
     Full LogitNormal(mu, sigma) parameters from prediction grids.

  3. mortality_posterior / diagnosis_posterior(icd_code)
     Exact structured posterior over model parameters (beta).
     See ParameterPosterior for the covariance decomposition.

When a patient has two complaints, predictions mix in probability space:
    P = 0.5 * sigmoid(mu_1) + 0.5 * sigmoid(mu_2)
This is the correct E[P] under complaint uncertainty (mixing in logit
space would be biased by Jensen's inequality).

Usage:
    from synth_ed_atlas import EdAtlas

    atlas = EdAtlas()

    # Point prediction (posterior mean)
    probs = atlas.predict(age=65, is_male=1, sctids=["271594007"])       # (930,)
    p_death = atlas.predict_death(age=65, is_male=1, sctids=["271594007"])

    # Predictive distribution parameters
    mu, sigma = atlas.predict_distribution(age=65, is_male=1, sctids=["271594007"])

    # Parameter posterior (exact structured covariance)
    post = atlas.mortality_posterior
    mu, sigma = post.predict(age=65, is_male=1, sctid="271594007")

    # Available codes
    atlas.icd_codes       # list[str], length 930
    atlas.diag_sctids     # list[str], length 435
    atlas.mort_sctids     # list[str], length 437
"""

from ._lookup import EdAtlas, ParameterPosterior

__all__ = ["EdAtlas", "ParameterPosterior"]
