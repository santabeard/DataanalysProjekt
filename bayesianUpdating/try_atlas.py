from synth_ed_atlas import EdAtlas

atlas = EdAtlas()
print(atlas)

# 65-year-old man with chest pain
print("P(death):", atlas.predict_death(age=65, is_male=1, sctids=["29857009"]))
print("diagnoses:", dict(zip(atlas.icd_codes, atlas.predict(65, 1, ["29857009"]).round(3))))

# The "trained model": posterior mean and covariance of the coefficients
post = atlas.mortality_posterior
print(post.beta_mean.shape)        # all coefficients
print(post.base_cov.shape)         # covariance of the 10 shared ones
mean, sd = post.beta_summary("29857009")
print(mean.round(2), sd.round(2))  # the 20 coefficients used for chest pain
