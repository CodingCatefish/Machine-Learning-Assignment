# INF2008 Machine Learning Assignment 2: Loan Default Prediction

## 📋 Project Overview

**Course:** INF2008 - Machine Learning (Stage 2)  
**Objective:** Advanced loan default prediction using CRISP-DM methodology, focusing on pipeline engineering, controlled experimentation, and business decision-making.

**Business Problem:** Predict loan defaults to minimize financial losses by identifying high-risk borrowers while avoiding unnecessary rejection of creditworthy applicants.

**Stage 2 Focus:** Transition from basic feasibility (Stage 1) to production-ready pipeline engineering, ablation studies, and actionable business policies.

## 👥 Team Structure & Roles

| Team Member | Role | Responsibilities |
|-------------|------|------------------|
| **YX** | Data Preparation & Pipeline Engineering | Robust scaling, encoding, missing value handling, data leakage prevention |
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
**Selection Criteria:** Highest ROC-AUC (0.7132) with superior recall performance

| Model | ROC-AUC | Recall@0.4 | False Positives | Training Time |
|-------|---------|------------|-----------------|---------------|
| **LightGBM** | **0.7132** | **81.52%** | 21,835 | ~8 seconds |
| XGBoost | 0.7120 | 81.55% | 22,071 | ~13 seconds |
| Random Forest | 0.6954 | 65.2% | 12,537 | ~55 seconds |

### Business Achievement
✅ **Exceeds 70% recall target** - Catches 81.5% of actual defaults  
✅ **Identifies 7,576 out of 9,293 defaults** (1,930 more than Random Forest)  
✅ **Accepts 21,835 false positives** (rejected good borrowers)

## 🔬 Ablation Study Results

**Champion Model:** LightGBM with controlled hyperparameter experiments

| Experiment | Parameter | Change | ROC-AUC Impact | Conclusion |
|------------|-----------|--------|----------------|------------|
| **E1** | n_estimators | 100→200 | +0.023% | Minimal improvement, baseline sufficient |
| E2 | num_leaves | 31→63 | -0.052% | Slight degradation, keep default |
| E3 | learning_rate | 0.1→0.05 | -0.192% | Worse performance, keep 0.1 |
| E4 | subsample | 0.8→0.9 | 0.000% | No change, keep 0.8 |

**Final Configuration:** n_estimators=200 (slight edge over baseline)

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
3. **Impact:** 81.5% recall vs. 50% at default threshold = 62% more defaults caught

## 📊 Key Performance Metrics

### Final Model Stability (Cross-Validation)
- **ROC-AUC:** 0.7132 ± 0.0041
- **F1-Score:** 0.4104 ± 0.0019
- **Recall:** 0.6496 ± 0.0020
- **Precision:** 0.2999 ± 0.0020

### Confusion Matrix at Business Threshold (0.4)
```
True Negatives: 19,710  |  False Positives: 21,835
False Negatives: 1,717   |  True Positives: 7,576
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
- imbalanced-learn>=0.11.0

### Usage
```bash
# Run the complete pipeline
python LGBM_version_MLAsg2.py

# Or run individual components
python machine_learning_assignment_2.py  # Random Forest version
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
├── machine_learning_assignment_2.py # Random Forest baseline
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

**Before (Random Forest at default threshold):**
- Catches 5,648 defaults (61% of total)
- Rejects 12,537 good borrowers
- **35% of defaults missed** = High financial risk

**After (LightGBM at optimized threshold):**
- Catches 7,576 defaults (81.5% of total)
- Rejects 21,835 good borrowers
- **18.5% of defaults missed** = Significantly reduced risk

**ROI Impact:** Prevents ~$38.6M in potential losses (1,930 additional defaults caught × $20K average loan)

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
