# Semiconductor-Manufacturing-Quality-Intelligence
An end-to-end machine learning system for semiconductor manufacturing quality prediction, anomaly detection, and process analysis.
# Semiconductor Manufacturing Quality Intelligence (SECOM)

Predicting which semiconductor chips will fail final test from process-sensor readings, and screening which sensors are most associated with failures.

> **Note on scope:** the SECOM dataset contains only a binary `Pass/Fail` label and 590 anonymous sensor readings. There is **no failure-cause label**, so the project predicts *whether* a chip fails and ranks *which sensors* are most associated with failure. It does not name a specific failure mode.

---

## Dataset

- **Source:** UCI SECOM dataset (`uci-secom.csv`)
- **Size:** 1,567 production runs x 590 sensor features, plus `Time` and `Pass/Fail`
- **Period:** January to December 2008
- **Class imbalance:** 104 failures (6.6%) vs 1,463 passes
- **Data quality:**
  - 116 sensors are constant (zero information)
  - 32 sensors have 40% or more missing values
  - Failure rate drifts over time (about 8% early on, about 3% in the last months)

An "always predict Pass" rule is already **93.4% accurate**, so accuracy is a misleading metric here.

---

## Phase 1: Original implementation (`main.ipynb`)

Everything was built from scratch with NumPy, following the CS229 style.

**1. Data cleaning**
- Dropped the `Time` column.
- Dropped columns with 40% or more missing values.
- Filled the remaining missing values with the column median.
- Re-labelled `Pass/Fail` from {-1, +1} to {0, 1} (1 = fail).

**2. Split and scaling**
- 70/30 random train/test split (`random_state=42`).
- `StandardScaler` fit on the training set only.

**3. Models**

| Model | Implementation | Notes |
|---|---|---|
| Logistic Regression | Batch gradient descent written by hand, L2 penalty, lr = 0.001 | Without regularisation it did not converge; a higher learning rate made it oscillate |
| Newton's Method | Hessian-based update | Failed with a *singular matrix* error |
| Gaussian Naive Bayes | Class priors, per-class mean/std, summed log-likelihoods | Higher recall, very poor precision |
| SVM | Data prepared with labels {-1, +1} | Not completed in the notebook |

**4. Original observations**
- Logistic Regression: about 33% recall but only about 18% precision. 
- Gaussian Naive Bayes: about 77% recall but very low precision and poor overall accuracy. Flags too many chips have failed, but actually good. Which is better comapred to logistic regression.
- Newton's method could not run.
- SVM 0 recall

---
