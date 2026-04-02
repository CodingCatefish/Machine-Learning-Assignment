# INF2008 Machine Learning Assignment 2: Loan Default Prediction

## 📋 Project Overview

**Course:** INF2008 - Machine Learning (Stage 2)

**Dataset**: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data

**Objective:** Advanced loan default prediction using CRISP-DM methodology, focusing on pipeline engineering, controlled experimentation, and business decision-making.

**Business Problem:** Predict loan defaults to minimize financial losses by identifying high-risk borrowers while avoiding unnecessary rejection of creditworthy applicants.

**Stage 2 Focus:** Transition from basic feasibility (Stage 1) to production-ready pipeline engineering, ablation studies, and actionable business policies.

## 👥 Team Structure & Roles

| Team Member | Role | Responsibilities |
|-------------|------|------------------|
| **Yi Xuan** | Data Preparation & Pipeline Engineering | Robust scaling, encoding, missing value handling, data leakage prevention |
| **Lionel** | Champion Model Selection | Cross-validation comparison of 2-3 algorithmic families |
| **Justin** | Ablations & Tuning | 4 controlled experiments on champion model with detailed logging |
| **Ezra** | Mechanical Failure Analysis | Row-level error inspection and technical fix proposals |
| **Jing Hai** | Decision Making & Consolidation | Business risk evaluation and threshold optimization |

## 🏗️ Technical Architecture

### Data Pipeline
- **Input:** Raw loan dataset (887,379 rows, 74 features)
- **Preprocessing:** Feature engineering, type casting, missing value imputation
- **Feature Selection:** Smart filtering of high-cardinality categorical features
- **Output:** Clean dataset (254,190 samples, 43 features)

### Model Pipeline
```
Raw Data → Imputation → Encoding/Scaling → Classifier → Predictions
```

**Key Components:**
- **Imputation:** Median for numeric, constant fill for categorical
- **Encoding:** One-hot encoding for categorical features
- **Scaling:** Robust scaling for numeric features
- **Class Balancing:** Cost-sensitive learning (scale_pos_weight) instead of synthetic resampling

## 🏆 Model Selection & Results

### Champion Model: LightGBM
**Selection Criteria:** Highest cross-validation ROC-AUC during Part B model selection

| Model | CV ROC-AUC | CV Recall (threshold=0.5) | Training Time |
|-------|------------|---------------------------|---------------|
| **LightGBM** | **0.7130** | **66.22%** | ~7 seconds |
| XGBoost | 0.7120 | 65.10% | ~12 seconds |
| Random Forest | 0.6891 | 37.89% | ~98 seconds |

**Important:** The table above is from 5-fold cross-validation during model selection. Threshold-specific business metrics for the final tuned LightGBM model are reported later in the README and are not directly comparable to these CV values.

### Business Achievement
✅ **Exceeds 70% recall target** - Catches 79.13% of actual defaults on the holdout set at threshold `0.4`  
✅ **Identifies 7,354 out of 9,293 defaults**  
✅ **Accepts 20,477 false positives** (rejected good borrowers)

## 🔬 Ablation & Tuning Strategy

**Champion Model: LightGBM**

To adhere to the constraints against massive computational brute-forcing, we avoided extensive grid searches. Instead, we performed a highly constrained, 30-iteration Bayesian Optimization scout (via `Optuna`) on the training folds to intelligently map the hyperparameter space. 

From this constrained search, we isolated the four most theoretically impactful parameters. We then formally evaluated these four parameters via a strict, one-at-a-time ablation study to justify their individual impact:

1. **`num_leaves` (Tree Complexity):** *Hypothesis* - Increasing this from 31 to 49 allows the model to capture more complex, non-linear financial relationships.
2. **`learning_rate` (Convergence Speed):** *Hypothesis* - A lower learning rate (0.058) helps the gradient boosting process converge more smoothly to a better global minimum.
3. **`min_child_samples` (Leaf-level Regularization):** *Hypothesis* - Increasing the minimum data per leaf from 20 to 33 prevents the model from overfitting to outlier borrower profiles.
4. **`scale_pos_weight` (Cost-Sensitive Penalty):** *Hypothesis* - Slightly lowering the penalty for the minority class finds a better optimal balance between Precision and Recall.

**Final Configuration:** The `scale_pos_weight=4.04` configuration was adopted as the final model because it delivered the best cross-validation F1-score in the ablation study.

## 🔍 Mechanical Failure Analysis

### False Positive Analysis (High-Confidence Errors)
**Pattern:** Model over-predicts defaults for borrowers with:
- High loan amounts ($20K-$35K range)
- Long credit history (200-350 months)
- Recent credit inquiries
- High utilization rates

**Technical Fixes Proposed:**
1. **Feature Engineering:** Add debt-to-income ratio calculations
2. **Threshold Calibration:** Implement borrower risk scoring bands
3. **Feature Importance:** Weight recent payment behavior more heavily

### False Negative Analysis (Missed Defaults)
**Pattern:** Model under-predicts defaults for:
- Short credit history (<100 months)
- High loan-to-value ratios
- Multiple recent derogatory marks

## 💼 Business Decision Making

### Error Cost Analysis
- **False Negative (Missed Default):** High cost - potential $20K+ loss per default
- **False Positive (Rejected Good Borrower):** Moderate cost - lost interest revenue

### Threshold Optimization
**Recommendation:** Shift threshold from 0.5 → 0.4

**Justification:**
1. **Asymmetric Costs:** Missing a default costs ~4.5× more than rejecting a good borrower
2. **Business Goal:** "Minimize loss by catching people who will default"
3. **Impact:** On the final tuned LightGBM model, threshold `0.4` achieved 79.13% recall on the holdout set while keeping false positives at 20,477.

## 📊 Key Performance Metrics

### Final Model Stability (Cross-Validation)
- **ROC-AUC:** 0.7127 ± 0.0040
- **F1-Score:** 0.4105 ± 0.0022
- **Recall:** 0.6159 ± 0.0031
- **Precision:** 0.3078 ± 0.0023

### Confusion Matrix at Business Threshold (0.4)
```
True Negatives: 21,068  |  False Positives: 20,477
False Negatives: 1,939   |  True Positives: 7,354
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

**Required Packages:**
- lightgbm>=4.0.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- polars>=0.20.0
- pyarrow>=15.0.0
- imbalanced-learn>=0.11.0

### Usage
```bash
# Run the complete pipeline
python LGBM_version_MLAsg2.py

# Or run individual components
python RandomForest_version_MLAsg2.py    # Random Forest version
python XGBOOST_version_MLAsg2.py         # XGBoost version
```

### Output Structure
1. **Part A:** Data preprocessing and pipeline validation
2. **Part B:** Model selection with cross-validation results
3. **Part C:** Ablation experiments and final model tuning
4. **Part D:** Mechanical failure analysis with specific examples
5. **Part E:** Business decision justification and threshold analysis

## 📁 Project Structure
```
├── LGBM_version_MLAsg2.py          # Champion LightGBM implementation
├── RandomForest_version_MLAsg2.py  # Random Forest baseline
├── XGBOOST_version_MLAsg2.py       # XGBoost alternative
├── loan.csv                        # Dataset (887K+ loan records)
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation
```

## 🎯 CRISP-DM Compliance

| Phase | Status | Deliverables |
|-------|--------|--------------|
| **Phase 1** | ✅ Complete | Data understanding, initial assessment |
| **Phase 2** | ✅ Complete | Data preparation, exploratory analysis |
| **Phase 3** | ✅ Complete | Feature engineering, pipeline construction |
| **Phase 4** | ✅ Complete | Model selection, cross-validation |
| **Phase 5** | ✅ Complete | Model evaluation, threshold optimization |
| **Phase 6** | ✅ Complete | Deployment recommendations, business justification |

## 📈 Business Impact

**Final tuned LightGBM at threshold 0.4:**
- Catches 7,354 defaults (79.13% of total)
- Rejects 20,477 good borrowers
- Misses 1,939 defaults (20.87% of total)

**Operational takeaway:** The tuned LightGBM model clears the 70% recall target and gives a documented trade-off between missed defaults and rejected good borrowers.

## 🤝 Individual Contributions

- **YX:** Pipeline architecture, feature engineering, data preprocessing
- **Lionel:** Model selection framework, cross-validation implementation
- **Justin:** Ablation experiments, hyperparameter tuning, performance logging
- **Ezra:** Error analysis, false positive/negative investigation, technical recommendations
- **Jing Hai:** Business analysis, threshold optimization, decision-making framework

## 📚 References

- CRISP-DM Methodology (Cross-Industry Standard Process for Data Mining)
- Scikit-learn Pipeline Documentation
- LightGBM Documentation
- Imbalanced-learn Library for Cost-sensitive Learning

---

**Submission Date:** April 3, 2026  
**Team:** INF2008 Group Project  
**Stage:** 2 (Advanced Pipeline Engineering & Business Decision Making)
