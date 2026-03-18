# XGBoost Thresholded Credit Risk Design

## Summary

Replace the current implicit Random Forest preference with an evidence-based champion selection flow that includes XGBoost as a candidate model family. The final champion must be supported by cross-validation and threshold analysis rather than hard-coded assumptions. For highly imbalanced credit-risk data, the decision threshold will be selected from probability outputs using portfolio expected value, while still visualizing the precision-recall trade-off for diagnostics.

## Current Context

- The project is a single-script workflow in [main.py](/c:/Users/jwooh/OneDrive/Documents/GitHub/Machine-Learning-Assignment/main.py).
- The data-preprocessing flow already produces a reusable `core_preprocessing_pipeline`.
- The model-selection section currently compares several estimators but the comments and downstream sections are still written around Random Forest as the assumed winner.
- The later evaluation block already uses `predict_proba()`, but it does not yet drive threshold optimization or expected-loss analysis.

## Goals

1. Add XGBoost as a valid candidate in model selection.
2. Declare a champion only if cross-validation and probability-threshold analysis justify it.
3. Use `predict_proba()` outputs to tune the decision threshold instead of relying on the default `0.50`.
4. Use a precision-recall curve because the target is imbalanced, even though it will be diagnostic rather than a hard gating rule.
5. Maximize portfolio expected value for approved loans using loan-level revenue and risk terms.
6. Provide a reusable threshold application function for future scoring.

## Non-Goals

- Building a separate LGD or EAD prediction model.
- Refactoring the entire notebook-style script into multiple modules.
- Introducing post-outcome leakage into the scoring pipeline.

## Design Decisions

### 1. Champion Selection Flow

- Add `XGBClassifier` as another candidate alongside the existing baseline models.
- Continue using the shared preprocessing pipeline so the comparison remains fair across model families.
- Rank models with cross-validation metrics suited to imbalanced classification:
  - `average_precision`
  - `f1`
  - `precision`
  - `recall`
  - `roc_auc`
  - `accuracy`
- Use cross-validation to identify the provisional best candidate.
- Fit that provisional champion on the training split and generate holdout probabilities with `predict_proba()[:, 1]`.
- Perform threshold search on those probabilities to verify whether the model still meets business needs under a custom cutoff.

### 2. Threshold Optimization Rule

- Build a precision-recall curve from holdout probabilities as a diagnostic view of the trade-off between precision and recall.
- Evaluate candidate thresholds across the holdout probabilities.
- For each threshold, treat loans with `PD < threshold` as approved and loans with `PD >= threshold` as rejected.
- Compute portfolio expected value for the approved set.
- Select the threshold that maximizes total portfolio expected value.
- Report precision, recall, approval rate, expected loss, expected interest, and expected value at the chosen threshold, but do not enforce a hard statistical floor.

### 3. Expected Value Formulation

- `PD`: predicted default probability from `predict_proba()`.
- `Probability_of_Paying`: `1 - PD`.
- `LGD`: one historical average estimated from past defaulted or charged-off training loans only.
- `EAD`: loan exposure at approval time, using `funded_amnt` when available and `loan_amnt` as fallback.
- `Expected_Interest_Rate`: a simple configurable business parameter applied to approved-loan exposure.
- `Expected_Interest_i`: `EAD_i x Expected_Interest_Rate`.

Expected loss component for an approved loan:

`EL_i = PD_i x historical_avg_LGD x EAD_i`

Expected value for an approved loan:

`EV_i = ((1 - PD_i) x Expected_Interest_i) - (PD_i x historical_avg_LGD x EAD_i)`

Portfolio expected value at a threshold:

`Portfolio_EV = sum(EV_i for approved loans)`

### 4. Leakage Controls

- Do not use realized loss variables such as `recoveries`, `total_rec_prncp`, or other post-outcome values as model inputs.
- These columns may be used once to estimate the historical average LGD from the training split.
- Threshold selection must consume only predicted probabilities, approval-time exposure values, and the explicit expected-interest assumption.

## Implementation Plan

### Model Selection Updates

- Import `XGBClassifier`.
- Extend the `model_configs` structure to include XGBoost base parameters and ablation settings.
- Add `average_precision` to the `scoring` list.
- Update printed summaries and ranking logic so the champion is not assumed to be Random Forest.

### Helper Functions

Add the following helpers to the script:

- `estimate_historical_average_lgd(...)`
  - Compute realized loss ratios on historical defaults only.
  - Clip ratios to `[0, 1]`.
  - Return one average LGD constant.
- `compute_expected_value(probabilities, lgd, ead, expected_interest_rate)`
  - Return per-loan expected values and the embedded expected-loss term.
- `select_threshold_by_expected_value(y_true, probabilities, ead, expected_interest_rate, lgd)`
  - Evaluate candidate thresholds using portfolio expected value.
  - Return the selected threshold and a threshold summary table.
- `apply_threshold(probabilities, threshold)`
  - Convert probabilities into final class labels for all future scoring.

### Evaluation Outputs

The script should print or plot:

- Cross-validated model leaderboard.
- Champion declaration with supporting metrics.
- Precision-recall curve for the final candidate.
- Threshold summary table showing precision, recall, approval rate, expected loss, expected interest, and expected value.
- Chosen threshold with rationale.
- Final metrics and confusion-matrix-style counts under the custom threshold.

### Ablation Handling

- Keep the existing ablation structure.
- Run ablations only for the selected champion model family.
- If XGBoost becomes champion, its ablations should be XGBoost-specific.

## Testing And Verification

Verification should confirm:

1. XGBoost appears in model selection and participates in cross-validation.
2. `predict_proba()` is used for threshold tuning.
3. The precision-recall curve is generated for the final candidate.
4. The threshold search maximizes portfolio expected value rather than enforcing a precision floor.
5. Expected value uses `((1 - PD) x Expected_Interest) - (PD x LGD x EAD)`.
6. Final class labels are produced through the reusable threshold function.

## Risks And Mitigations

- XGBoost may not be installed.
  - Fail with a clear import or dependency message instead of silently skipping it.
- Threshold results may be unstable if the holdout set is small.
  - Report threshold metrics explicitly and keep the rule transparent.
- Historical LGD estimation may be noisy.
  - Use a clipped average from defaults only and document the assumption.
- The selected threshold may be sensitive to the revenue assumption.
  - Keep the expected-interest parameter explicit and report it with the chosen threshold.

## User-Approved Decisions

- XGBoost is added as a candidate rather than forced as champion.
- Champion status requires both cross-validation support and threshold analysis support.
- Thresholding uses probability outputs from `predict_proba()`.
- Precision-recall analysis is retained as a diagnostic visualization, not a hard threshold rule.
- There is no hard precision floor.
- Threshold selection is driven by portfolio expected value.
- Expected value follows `((1 - PD) x Expected_Interest) - (PD x LGD x EAD)`.
- `LGD` is estimated as a historical average from defaulted loans.
- A simple expected-interest parameter is part of the decision rule.
