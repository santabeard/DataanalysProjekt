# Evaluation Metrics: Questions and Mathematical Definitions

This document defines the metrics used for posterior-aware evaluation of:

- **Diagnosis risk**: multi-label diagnosis-code presence linked to the index ED visit.
- **Patient risk**: 30-day mortality from the patient's first ED visit.

Let:

- $s=1,\dots,S$ index posterior draws.
- $i$ index patients.
- $k$ index diagnosis codes.
- $y\in\{0,1\}$ be an observed binary label.
- $p^{(s)}\in[0,1]$ be a posterior risk draw.
- $\mathbf{1}[\cdot]$ be the indicator function.
- $\operatorname{logit}(p)=\log(p/(1-p))$.

For any binary evaluation labels $y$ and scores $p$:

$$
\operatorname{AUC}(y,p)=
\frac{
\sum_{a:y_a=1}\sum_{b:y_b=0}
\left[
\mathbf{1}(p_a>p_b)+\tfrac12\mathbf{1}(p_a=p_b)
\right]
}{n_1 n_0}
$$

where $n_1$ is the number of positives and $n_0$ is the number of negatives.

| Metric | Performance question | Mathematical definition |
|---|---|---|
| **Per-diagnosis ROC-AUC** | For diagnosis $k$, can the model rank patients with $k$ above patients without $k$? | $\operatorname{AUC}_{k,s}=\operatorname{AUC}(y_{\cdot k},p_{\cdot k}^{(s)})$, where $y_{\cdot k}=(y_{1k},\dots,y_{nk})$. |
| **Diagnosis micro-AUC** | Across all patient–diagnosis pairs, are positives ranked above negatives? Common diagnoses dominate. | Let $\mathcal A_s=\{(i,k):p_{ik}^{(s)}\text{ finite}\}$. Then $\operatorname{AUC}^{\text{micro}}_s=\operatorname{AUC}(\{y_{ik}\}_{(i,k)\in\mathcal A_s},\{p_{ik}^{(s)}\}_{(i,k)\in\mathcal A_s})$. |
| **Diagnosis macro-AUC** | For an average diagnosis code, can the model separate positives from negatives? Codes weighted equally. | Let $\mathcal E=\{k:\sum_i y_{ik}\ge m,\ n-\sum_i y_{ik}\ge m\}$. Then $\operatorname{AUC}^{\text{macro}}_s=\frac{1}{\lvert\mathcal E\rvert}\sum_{k\in\mathcal E}\operatorname{AUC}_{k,s}$. |
| **Patient 30-day mortality AUC** | At the first ED visit, can the model rank 30-day deaths above survivors? | $\operatorname{AUC}^{\text{death}}_s=\operatorname{AUC}(y_i,p_i^{(s)})$, where $y_i=1$ if patient $i$ dies within 30 days. |
| **Patient micro-AUC including mortality** | Across all patient-level outcomes, diagnosis-code presence plus 30-day mortality, is a randomly chosen positive patient–outcome pair ranked above a randomly chosen negative patient–outcome pair? | Let $\tilde y_{i,k}=y_{ik}$ for $k=1,\dots,K$ and $\tilde y_{i,K+1}=y_i^{\text{death}}$. Let $\tilde p_{i,k}^{(s)}=p_{ik}^{(s)}$ for $k=1,\dots,K$ and $\tilde p_{i,K+1}^{(s)}=p_i^{\text{death},(s)}$. With $\tilde{\mathcal A}_s=\{(i,j):\tilde p_{i,j}^{(s)}\text{ finite}\}$, $\operatorname{AUC}^{\text{patient-micro+death}}_s=\operatorname{AUC}(\{\tilde y_{i,j}\}_{(i,j)\in\tilde{\mathcal A}_s},\{\tilde p_{i,j}^{(s)}\}_{(i,j)\in\tilde{\mathcal A}_s})$. |
| **Negative log-likelihood / log score** | Are predicted probabilities consistent with observed events, penalising confident mistakes? | $\ell_{r,s}=-[y_r\log p_r^{(s)}+(1-y_r)\log(1-p_r^{(s)})]$; $\operatorname{NLL}_s=\frac{1}{R}\sum_{r=1}^R\ell_{r,s}$. |
| **Brier score** | How close are probabilities to observed 0/1 outcomes? | $\operatorname{BS}_s=\frac{1}{R}\sum_{r=1}^R(p_r^{(s)}-y_r)^2$. |
| **Brier skill score** | How much better than a prevalence-only baseline? | $\bar y=R^{-1}\sum_r y_r$, $\operatorname{BS}_{\text{base}}=\bar y(1-\bar y)$, $\operatorname{BSS}_s=1-\operatorname{BS}_s/\operatorname{BS}_{\text{base}}$. |
| **O:E ratio** | Did the model predict the right total number of events? | $\operatorname{OE}_s=\dfrac{\sum_r y_r}{\sum_r p_r^{(s)}}$. Ideal value: $1$. |
| **Calibration slope** | Are risk predictions too extreme or too flat? | Fit $\operatorname{logit}\Pr(y_r=1)=\alpha+\beta\operatorname{logit}(\bar p_r)$, where $\bar p_r=S^{-1}\sum_s p_r^{(s)}$. Report $\beta$; ideal value: $1$. |
| **Calibration intercept** | Are risks systematically too high or too low? | In the same calibration model, report $\alpha$; ideal value: $0$. |
| **Decile calibration** | Among similar-risk patients, do observed rates match predicted risks? | Bin units into deciles $Q_b$ of $\bar p_r$. $\operatorname{obs}_b=\lvert Q_b \rvert^{-1}\sum_{r\in Q_b}y_r$, $\operatorname{pred}_b=\lvert Q_b \rvert^{-1}\sum_{r\in Q_b}\bar p_r$. |
| **N, events, prevalence** | Are there enough events/non-events for the metric to be stable? | $R=\lvert\{r\}\rvert$, $E=\sum_r y_r$, $\hat q=E/R$. For diagnosis $k$: $E_k=\sum_i y_{ik}$, $\hat q_k=E_k/n$. |
| **Posterior interval** | How uncertain is the metric given posterior uncertainty? | For metric values $\{M_s\}_{s=1}^S$, report $\operatorname{median}(M_s)$ and $[Q_{0.025}(M_s),Q_{0.975}(M_s)]$. |
| **Patient bootstrap CI** | How sensitive is the metric to the particular test-patient sample? | For bootstrap samples $b=1,\dots,B$, resample patients, recompute $M^{(b)}$, and report $[Q_{0.025}(M^{(b)}),Q_{0.975}(M^{(b)})]$. |
| **Subgroup metrics** | Does performance differ by hospital, age, sex, complaint, or care level? | For subgroup $g$: $\operatorname{AUC}_{g,s}=\operatorname{AUC}(\{y_r:g_r=g\},\{p_r^{(s)}:g_r=g\})$. The same restriction applies to NLL, Brier, O:E and calibration. |

If $n_1=0$ or $n_0=0$, AUC is undefined and should be skipped.

For diagnosis evaluation, use **macro-AUC as the primary discrimination metric** and **micro-AUC as secondary**, because micro-AUC is weighted by diagnosis frequency. **Patient micro-AUC including mortality** pools diagnosis-code outcomes and 30-day mortality as additional columns for the same patients; it is distinct from diagnosis-only micro-AUC and death-only AUC. Mortality contributes one outcome column per patient, while diagnosis codes contribute $K$ columns. AUC measures discrimination only; calibration and probability quality are answered by NLL, Brier score, O:E ratio, calibration slope/intercept and calibration curves.

The evaluator writes a CSV with one row per diagnosis plus `macro` and `micro`
aggregates; a binary mortality target has one row. ROC-AUC, Brier score, NLL and
O:E ratio are summarized by posterior median and 95% credible interval.
Calibration slope and intercept are fitted using posterior-mean risks. It also
saves a calibration plot and a ROC plot (micro/macro curves for multi-label
diagnoses; a binary ROC curve for mortality) beside the CSV.
