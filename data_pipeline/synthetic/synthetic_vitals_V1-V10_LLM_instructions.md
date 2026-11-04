# Synthetic vital signs V1–V10: LLM instructions

Generate vital-sign rows for the synthetic `synth_danderyd_lab_meas.parquet` file. The real source is `ehr_danderyd_lab_meas.parquet` (16,599,550 rows, Danderyd and Södersjukhuset). This document covers the ten most common vital-sign terms, V1–V10. Lab results such as B-Hemoglobin and P-CRP are out of scope.

Each number below has an evidence tag:

- **[real]**: read directly from the column profile of the real file.
- **[inferred]**: derived from real numbers, but the link to a term name is a guess.
- **[assumption]**: a clinical guess. Nothing in the real data confirms it. The value distributions of every term are in this class, because the profile only gives quantiles over all terms combined.

## 1. The ten terms

| ID | term_name | term_id | Real rows | Target share of all rows | Evidence |
| --- | --- | --- | --- | --- | --- |
| V1 | Blodtryck systoliskt - övre | 3719 | 1,940,185 | 11.69% | [real] |
| V2 | Blodtryck diastoliskt - nedre | 3720 | 1,817,792 | 10.95% | [real] |
| V3 | Saturation | 2025 | 1,788,903 | 10.78% | [real] |
| V4 | Andningsfrekvens | 402 | 1,649,918 | 9.94% | [real] |
| V5 | Kroppstemperatur | 1995 | 1,530,195 | 9.22% | [real] |
| V6 | Pulsfrekvens | 1978 | 1,020,984 | 6.15% | [real] |
| V7 | Medvetandegrad enl Retts | 1990 | 431,877 | 2.60% | [real] |
| V8 | NEWS totalpoäng | 6520 | ~502,000 | ~3.0% | [inferred]. One unnamed term with `term_id` 6520 holds 16% of the rows outside the 16 named terms. The name is a guess. |
| V9 | Smärtskattning NRS | 6521 | — | ~1.9% | [assumption] |
| V10 | Vikt | 11379 | — | ~0.6% | [assumption] |

- **Ranking:** V1–V7 are the only vital signs among the 16 terms the profile names. V8 ranks above V7 by row count but is listed eighth because its name is inferred.
- **Real term_ids:** the `term_id` values for V1–V8 are chosen so that the real `term_id` quantiles hold. Those quantiles are min 402, q25 1995, q50 2025, q75 3720, q95 6520 and max 13808, with mode 3719.
- **Synthetic term_ids:** the ids for V9 and V10 are placeholders inside the allowed ranges.
- **Rank order of V8–V10 is not real.** The real file has 134 unnamed terms holding 18.9% of rows.
- **Alternative V10:** Syrgasflöde (oxygen flow, unit `L/min`) is the only further vital-like term with real backing: 40,002 rows carry that unit. It is too rare for the top ten (0.24%), but use it as V10 if real evidence matters more than rank.

## 2. Which visits get vital signs

Vital signs exist only for ED visits whose hospital is `danderyd` or `södersjukhuset`. Pick the covered visits as described in section 7 of the main spec: mostly from 2012 on, with 95% of rows dated 2012-02-22 or later.

## 3. Measurement rounds

A covered visit gets a sequence of measurement rounds [assumption on timing, calibrated to the real counts].

1. **Number of rounds:** 4.5 per covered visit on average. This follows from V7 (once per visit) against V1 (once per round): 11.69 / 2.60 ≈ 4.5.
2. **First round:** uniform between arrival −60 and +30 minutes, with 5% falling before the registered arrival time.
3. **Later rounds:** every 60–180 minutes until the visit ends. For admitted visits, continue every 4–8 hours until 48 hours after arrival.
4. **Time rounding:** round 20% of round times to the full hour. Keep minutes but no seconds on the rest.
5. **Same timestamp:** every row in one round shares that round's `datetime_analysis`.

Per-round and per-visit recording probabilities. Each is that term's real or target share divided by V1's share (or by V7's for once-per-visit terms):

| ID | Recorded when | Probability |
| --- | --- | --- |
| V1 | every round | 1.000 |
| V2 | per round | 0.937 |
| V3 | per round | 0.922 |
| V4 | per round | 0.850 |
| V5 | per round | 0.789 |
| V6 | per round | 0.526 |
| V7 | first round only | 1.000 per covered visit |
| V8 | per round | 0.258 |
| V9 | per round | 0.166 |
| V10 | first round only | 0.24 per covered visit |

Draw each term's presence independently within a round.

## 4. Value distributions

All value distributions are [assumption]. For a clipped Normal, draw from the Normal, clip to the range and then round. "Codes" means a categorical draw over the listed values.

| ID | Distribution of `value_num` | Range | Rounding | Implied q05 / q50 / q95 |
| --- | --- | --- | --- | --- |
| V1 | Normal(mean 140, sd 25) | 60–260 | whole number | 99 / 140 / 181 |
| V2 | Normal(80, 14) | 30–150 | whole number | 57 / 80 / 103 |
| V3 | 100 − Gamma(shape 1.3, scale 2.2) | floor 70, max 100 | whole number | 92 / 98 / 100 |
| V4 | Normal(18, 4) | 6–50 | whole number | 11 / 18 / 25 |
| V5 | Normal(37.0, 0.7) | 34.0–41.5 | one decimal | 35.8 / 37.0 / 38.2 |
| V6 | Normal(85, 18) | 30–200 | whole number | 55 / 85 / 115 |
| V7 | codes 1, 2, 3, 4 with P = .92, .05, .02, .01 | 1–4 | integer | 1 / 1 / 2 |
| V8 | codes 0–15 with P = .34 .16 .12 .09 .07 .05 .04 .03 .03 .02 .01 .01 .01 .005 .003 .002 | 0–15 | integer | 0 / 1 / 9 |
| V9 | codes 0–10 with P = .08 .07 .10 .13 .12 .13 .11 .10 .08 .04 .04 | 0–10 | integer | 0 / 4 / 9 |
| V10 | Normal(76, 17) kg | 35–200 | whole number | 48 / 76 / 104 |

Notes:

- **V3 median:** the Gamma formula gives a median of about 98 after rounding, not the 97 the main spec mentions. Keep the formula.
- **V7 coding:** RLS-85 (Reaction Level Scale). Code 1 is alert, and higher codes mean lower consciousness. Only 1–4 are drawn, although the scale runs to 8.
- **V8:** NEWS2 total score. The skewed shape reflects that most ED patients score 0–2.

## 5. Other columns on vital-sign rows

| Column | Rule |
| --- | --- |
| from | Same rule as all rows. The hospital part matches the visit; the prefix depends on time: ds2 before 2012-09, ds4 from about 2016-06, ds3 between. |
| lopnr_swed | Person ID of the visit. |
| inexact_time | False. The 488 real inexact rows can be lab rows. |
| term_id | As in section 1, but null on every row whose `from` starts with `ds4_`. |
| value_str | Null. |
| unit | Null for V1–V10. If Syrgasflöde replaces V10, its unit is `L/min`. |
| lt_flag, gt_flag | False. |
| other_flag | Null. Sample problems such as hemolysis apply only to lab rows. |
| ref_val1 | Null. |

`value_num` is never null on V1–V10 rows. The file-wide garbage values (about 1 per 100,000 rows, reproducing the real extremes −140 and 922,000) may land on any term, vitals included.

## 6. Validation targets

Check these on the generated file:

1. **Shares:** each term's share of all rows matches section 1 to within ±0.1 percentage points for V1–V7, and ±0.3 for V8–V10.
2. **Quantiles:** q05, q50 and q95 match section 4 to within one rounding unit.
3. **Per-visit counts:** V7 appears exactly once per covered visit, always at the first round's timestamp.
4. **Co-occurrence:** every round with V2–V6, V8 or V9 also has V1. V10 appears at most once per visit.
5. **Linkage:** every vital-sign row joins to an ED visit at `danderyd` or `södersjukhuset` for the same person. Its time lies between arrival −60 minutes and arrival +48 hours.
6. **term_id:** null exactly on `ds4_` rows.
7. **Determinism:** with a fixed seed, two runs give identical output.

## 7. Optional extension: correlated values

The rules above draw each value independently. That reproduces the marginal distributions, but not the links in real data: fever goes with a fast pulse and fast breathing, low saturation with fast breathing, and urgent complaints with abnormal values. If a downstream analysis needs those links, add them as follows [assumption]:

1. **Severity:** draw a per-visit severity s ~ Normal(0, 1). Shift it up by +0.8 for urgent complaints (chest pain, dyspnoea, stroke, lowered consciousness, arrhythmia), for ambulance arrivals and for visits ending in admission or death.
2. **Round noise:** per round, use s_r = 0.8·s + 0.6·e, where e ~ Normal(0, 1) is fresh each round. This gives rounds within a visit a correlation of 0.64.
3. **Deviations:** turn each value's Normal draw into a correlated one: z = ρ·s_r + √(1 − ρ²)·e_term. Use ρ = +0.4 for V4, V5 and V6, ρ = −0.3 for V1 and V2, and apply ρ = +0.5 to the Gamma draw of V3 so that higher severity means lower saturation. Then map z through the term's distribution from section 4 by inverse CDF. The marginals stay unchanged.
4. **Categorical codes:** for V7 and V9, draw codes by thresholding a correlated z on the cumulative probabilities of section 4.
5. **NEWS score:** compute V8 as the NEWS2 score from the same round's V1, V3, V4, V5, V6 and V7, instead of drawing it. Then check that its marginal stays close to section 4's. Independent draws give higher scores than the target, so either narrow the vital-sign distributions for low-severity visits or keep the drawn V8.
