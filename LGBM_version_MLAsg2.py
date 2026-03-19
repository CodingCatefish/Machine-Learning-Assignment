# Base Imports
import kagglehub
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import time
import warnings
warnings.filterwarnings('ignore')  # Suppress LightGBM and sklearn warnings for cleaner output

# Imports for Part 2

# Data Prep Phase - YX
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

# Model Selection Phase - Lionel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.base import clone
from tqdm import tqdm

# Ablations and tuning phase - Justin

# Mechanical failure analysis - Ezra

# Decision Making and consolidation - Jing Hai

df = pl.read_csv("loan.csv")

print(f"Data loaded: shape={df.shape}")

# Yixuan - Data Preperation
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import precision_recall_curve

print("Part A: Data Prep & Pipeline Engineering")

# --- Initial Setup & Target Definition (Polars) ---
# Define target features
df = df.filter(
    pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"])
).with_columns(
    pl.when(pl.col("loan_status") == "Fully Paid")
    .then(0)
    .otherwise(1)
    .alias("target")
)

# 1. Convert Polars DataFrame to Pandas (Crucial for scikit-learn compatibility)
df_pd = df.to_pandas()

# --- NEW: Feature Engineering & Type Fixing ---
print("Executing feature engineering and data type corrections...")

# A. Calculate Credit History Length (Months)
if 'issue_d' in df_pd.columns and 'earliest_cr_line' in df_pd.columns:
    df_pd['issue_d_parsed'] = pd.to_datetime(df_pd['issue_d'], format='%b-%Y', errors='coerce')
    df_pd['earliest_cr_line_parsed'] = pd.to_datetime(df_pd['earliest_cr_line'], format='%b-%Y', errors='coerce')

    # Calculate months between first credit line and loan issue date
    df_pd['credit_hist_months'] = (df_pd['issue_d_parsed'] - df_pd['earliest_cr_line_parsed']).dt.days / 30

    # Drop the temporary parsed columns
    df_pd = df_pd.drop(columns=['issue_d_parsed', 'earliest_cr_line_parsed'])

# B. Fix Percentages (if they exist as strings like "15.5%")
for col in ['int_rate', 'revol_util']:
    if col in df_pd.columns and df_pd[col].dtype == 'object':
        df_pd[col] = df_pd[col].astype(str).str.rstrip('%').astype('float')

# C. Force-cast High-Value Financials to Numeric
numeric_rescue_cols = [
    'tot_cur_bal', 'total_rev_hi_lim', 'total_bal_il',
    'il_util', 'max_bal_bc', 'all_util'
]
for col in numeric_rescue_cols:
    if col in df_pd.columns:
        df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

# --- Feature Selection ---
# 2. Select strict Features (X) and Target (y)
target_col = "target"
drop_cols = [
    target_col, "loan_status", "id", "member_id", "url", "desc", "emp_title",
    "title", "zip_code", "funded_amnt", "funded_amnt_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "out_prncp", "out_prncp_inv",
    "debt_settlement_flag", "hardship_flag", "pymnt_plan", "grade", "sub_grade",
    "issue_d", "earliest_cr_line" # Original dates dropped to avoid cardinality issues
]

cols_to_drop = [c for c in drop_cols if c in df_pd.columns]
X = df_pd.drop(columns=cols_to_drop)
y = df_pd[target_col]

# 3. Smart filtering of Categorical columns (Dropping high-cardinality)
cat_cols_all = X.select_dtypes(include=['object', 'string']).columns
safe_cat_cols = [col for col in cat_cols_all if X[col].nunique() < 50]
unsafe_cat_cols = [col for col in cat_cols_all if col not in safe_cat_cols]

print(f"Dropping high-cardinality features: {unsafe_cat_cols}")
X = X.drop(columns=unsafe_cat_cols)

# Define final numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = safe_cat_cols

# 4. Split and Stratify Data
print(f"Splitting data (Input shape: {X.shape})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Core Pipeline Assembly ---
# 5. Pipeline Step 1: Imputation
imputer_step = ColumnTransformer(
    transformers=[
        (
            'num_imputer',
            SimpleImputer(strategy='median', keep_empty_features=True),
            num_cols
        ),
        (
            'cat_imputer',
            SimpleImputer(
                strategy='constant',
                fill_value='Missing',
                keep_empty_features=True
            ),
            cat_cols
        )
    ],
    remainder='drop'
)

# 6. Pipeline Step 2: Scaling & Encoding
# IMPROVED: Removed aggressive resampling (undersampling + SMOTE-NC) which caused over-prediction of defaults.
# Now using class_weight='balanced' in the classifier instead for cost-sensitive learning.
encoder_scaler_step = ColumnTransformer(
    transformers=[
        ('num_scaler', RobustScaler(), list(range(len(num_cols)))),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(range(len(num_cols), len(num_cols) + len(cat_cols))))
    ],
    remainder='passthrough'
)

# 7. Assemble the Core Preprocessing Pipeline (SIMPLIFIED: removed resampling steps)
core_preprocessing_pipeline = ImbPipeline(steps=[
    ('imputation', imputer_step),
    ('scaling_encoding', encoder_scaler_step)
])

print("Data Prep and Pipeline Engineering is complete. Ready for handoff.")

# This section is used to generate visuals for Data Preparation and Pipeline Engineering phase.
print("--- Generating Visuals for Data Preparation and Pipeline Engineering phase ---")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# This section is used to generate visuals for Data Preparation and Pipeline Engineering phase.
print("--- Generating Visuals for Data Preparation and Pipeline Engineering phase ---")

# IMPROVED: Show original class distribution
# Using class_weight='balanced' in the classifier handles the imbalance without distorting training data
class_counts = y_train.value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, hue=class_counts.index, palette=["#1f77b4", "#ff7f0e"], legend=False)
ax.set_title(f"Training Data Distribution\nUsing class_weight='balanced' for cost-sensitive learning\nTotal Rows: {len(y_train):,}")
ax.bar_label(ax.containers[0], fmt='%d')
ax.set_ylabel("Count")
ax.set_xlabel("Loan Status")

plt.tight_layout()
plt.show()

print("Note: Using class_weight='balanced' instead of aggressive resampling (undersampling+SMOTE)")
print("This prevents overprediction of defaults on real-world test data.")

"""###Lionel - Model Selection"""

print("Part B: Model Selection")

# 1. Models: LightGBM, Bernoulli Naive Bayes, Logistic Regression
# IMPROVED: Added scale_pos_weight for LightGBM to handle class imbalance without distorting training data
# LightGBM is a gradient boosting model that often outperforms other models on tabular data

# Calculate scale_pos_weight for LightGBM (inverse of class ratio)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

models = {
     "LightGBM": LGBMClassifier(
         n_estimators=100,
         num_leaves=31,
         max_depth=7,
         learning_rate=0.1,
         subsample=0.8,
         colsample_bytree=0.8,
         scale_pos_weight=scale_pos_weight,  # ← Handle class imbalance
         random_state=42,
         n_jobs=1,
         verbose=-1  # Suppress LightGBM training output
     ),

    "BernoulliNaiveBayes": BernoulliNB(),

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # ← Handle class imbalance
        random_state=42
    )
}

# 2. K-Fold Strategy

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 3. Scoring: ROC AUC, F1, Precision, Recall, Accuracy

scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]

# 4. Cross Validation

results = []
for name, model in tqdm(models.items(), desc="Model Training"):
    start = time.time()
    print(f"\nStarting {name}")

    full_pipeline = ImbPipeline(
        steps=core_preprocessing_pipeline.steps + [("classifier", model)]
    )

    cv_scores = cross_validate(
        full_pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    elapsedTime = (time.time()) - start
    print(f"{name} finished in {elapsedTime:.2f} seconds")

    results.append({
        "Model": name,
        "ROC_AUC": cv_scores["test_roc_auc"].mean(),
        "F1": cv_scores["test_f1"].mean(),
        "Precision": cv_scores["test_precision"].mean(),
        "Recall": cv_scores["test_recall"].mean(),
        "Accuracy": cv_scores["test_accuracy"].mean()
    })

results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)

print(results_df)

# # 5. Bar Chart
# results_df.set_index("Model")[["ROC_AUC","F1"]].plot(kind="bar")
# plt.title("Model Comparison")
# plt.ylabel("Score")
# plt.show()

'''
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from tqdm import tqdm
import time
import pandas as pd

print("Part B: Model Selection")

# 1. Models: Random Forest, Bernoulli Naive Bayes, Logistic Regression
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, # Lowered slightly for faster CV, can tune in Part C.
        max_depth=None,
        n_jobs=1,         # Fix here : Set to 1 to prevent CPU thrashing.
        random_state=42
    ),
    "BernoulliNaiveBayes": BernoulliNB(),

    # Fix here : Replaced SVC with LogisticRegression for a linear family baseline.
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

# 2. K-Fold Strategy
kfold = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# 3. Scoring Metrics
scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]

# 4. Cross Validation Loop
results = []
for name, model in tqdm(models.items(), desc="Model Training"):
    start = time.time()
    print(f"\nStarting {name}...")

    # Fix here : Safely construct the full pipeline without in-place mutation using ImbPipeline.
    full_pipeline = ImbPipeline(
        steps=core_preprocessing_pipeline.steps + [("classifier", model)]
    )

    # Run the K-Fold CV
    cv_scores = cross_validate(
        full_pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1 # Parallelize across the folds. Only use n_jobs=-1 in CV.
    )

    elapsedTime = time.time() - start
    print(f"{name} finished in {elapsedTime:.2f} seconds")

    results.append({
        "Model": name,
        "ROC_AUC": cv_scores["test_roc_auc"].mean(),
        "F1": cv_scores["test_f1"].mean(),
        "Precision": cv_scores["test_precision"].mean(),
        "Recall": cv_scores["test_recall"].mean(),
        "Accuracy": cv_scores["test_accuracy"].mean()
    })

# 5. Display Results
results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
print("\n=== Model Selection Results ===")
print(results_df.to_string(index=False))

'''
# Total Takes about 20 minutes to execute. I fucking hate shift-enter :)

"""### Justin ー Ablation & Tuning"""

# Set up selected baseline champion (LightGBM)
# IMPROVED: Using scale_pos_weight for handling class imbalance
lgbm_baseline_params = {
    "n_estimators": 100,
    "num_leaves": 31,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1
}

# Four experiments; each changes only one parameter from the baseline
ablation_experiments = [
    {
        "Experiment": "E1_n_estimators_200",
        "Changed_Parameter": "n_estimators",
        "Old_Value": 100,
        "New_Value": 200,
        "Params": {**lgbm_baseline_params, "n_estimators": 200}
    },
    {
        "Experiment": "E2_num_leaves_63",
        "Changed_Parameter": "num_leaves",
        "Old_Value": 31,
        "New_Value": 63,
        "Params": {**lgbm_baseline_params, "num_leaves": 63}
    },
    {
        "Experiment": "E3_learning_rate_0.05",
        "Changed_Parameter": "learning_rate",
        "Old_Value": 0.1,
        "New_Value": 0.05,
        "Params": {**lgbm_baseline_params, "learning_rate": 0.05}
    },
    {
        "Experiment": "E4_subsample_0.9",
        "Changed_Parameter": "subsample",
        "Old_Value": 0.8,
        "New_Value": 0.9,
        "Params": {**lgbm_baseline_params, "subsample": 0.9}
    }
]

ablation_results = []

def build_cv_row(experiment_name, changed_param, old_value, new_value, cv_scores, elapsed):
    return {
        "Experiment": experiment_name,
        "Changed_Parameter": changed_param,
        "Old_Value": old_value,
        "New_Value": new_value,

        "ROC_AUC": cv_scores["test_roc_auc"].mean(),
        "ROC_AUC_STD": cv_scores["test_roc_auc"].std(ddof=1),

        "F1": cv_scores["test_f1"].mean(),
        "F1_STD": cv_scores["test_f1"].std(ddof=1),

        "Precision": cv_scores["test_precision"].mean(),
        "Precision_STD": cv_scores["test_precision"].std(ddof=1),

        "Recall": cv_scores["test_recall"].mean(),
        "Recall_STD": cv_scores["test_recall"].std(ddof=1),

        "Accuracy": cv_scores["test_accuracy"].mean(),
        "Accuracy_STD": cv_scores["test_accuracy"].std(ddof=1),

        "Runtime_sec": elapsed
    }

# Run baseline first
print("\nRunning baseline LightGBM...")
baseline_start = time.time()

baseline_model = LGBMClassifier(**lgbm_baseline_params)
baseline_pipeline = ImbPipeline(
    steps=core_preprocessing_pipeline.steps + [("classifier", baseline_model)]
)

baseline_cv_scores = cross_validate(
    baseline_pipeline,
    X_train,
    y_train,
    cv=kfold,
    scoring=scoring,
    n_jobs=-1
)

baseline_elapsed = time.time() - baseline_start

baseline_result = build_cv_row(
    experiment_name="Baseline",
    changed_param="None",
    old_value="-",
    new_value="-",
    cv_scores=baseline_cv_scores,
    elapsed=baseline_elapsed
)
ablation_results.append(baseline_result)

# Run the 4 controlled experiments
for exp in ablation_experiments:
    print(f"\nRunning {exp['Experiment']}...")
    start = time.time()

    model = LGBMClassifier(**exp["Params"])
    pipeline = ImbPipeline(
        steps=core_preprocessing_pipeline.steps + [("classifier", model)]
    )

    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    elapsed = time.time() - start

    ablation_results.append(
        build_cv_row(
            experiment_name=exp["Experiment"],
            changed_param=exp["Changed_Parameter"],
            old_value=exp["Old_Value"],
            new_value=exp["New_Value"],
            cv_scores=cv_scores,
            elapsed=elapsed
        )
    )

# Build ablation log
ablation_log_df = pd.DataFrame(ablation_results)

# Add deltas against baseline (means only)
baseline_row = ablation_log_df.iloc[0]
for metric in ["ROC_AUC", "F1", "Precision", "Recall", "Accuracy"]:
    ablation_log_df[f"Delta_{metric}"] = ablation_log_df[metric] - baseline_row[metric]

# Sort experiments by ROC_AUC mean for quick comparison
ablation_sorted_df = ablation_log_df.sort_values(by="ROC_AUC", ascending=False)

print("\n" + "="*150)
print("PART C: ABLATION & TUNING - LightGBM Hyperparameter Experiments")
print("="*150)
print("\n=== LightGBM Ablation Log ===")
print(ablation_log_df.to_string(index=False))

print("\n=== Ranked by ROC_AUC ===")
print(ablation_sorted_df.to_string(index=False))

# Parameter selection.
best_row = ablation_sorted_df.iloc[0]
best_experiment_name = best_row["Experiment"]

if best_experiment_name == "Baseline":
    final_model_params = lgbm_baseline_params
else:
    final_model_params = next(
        exp["Params"] for exp in ablation_experiments
        if exp["Experiment"] == best_experiment_name
    )

print(f"\n✓ Best Ablation Experiment: {best_experiment_name}")
print(f"  Improvement in ROC_AUC: +{best_row['Delta_ROC_AUC']*100:.3f}%")

# Cross validation with selected parameter.
final_model = LGBMClassifier(**final_model_params)
final_pipeline = ImbPipeline(
    steps=core_preprocessing_pipeline.steps + [("classifier", final_model)]
)

final_cv_scores = cross_validate(
    final_pipeline,
    X_train,
    y_train,
    cv=kfold,
    scoring=scoring,
    n_jobs=-1
)

final_stability_metrics_df = pd.DataFrame({
    "Metric": ["ROC_AUC", "F1", "Precision", "Recall", "Accuracy"],
    "CV Mean": [
        final_cv_scores["test_roc_auc"].mean(),
        final_cv_scores["test_f1"].mean(),
        final_cv_scores["test_precision"].mean(),
        final_cv_scores["test_recall"].mean(),
        final_cv_scores["test_accuracy"].mean()
    ],
    "CV Std Dev": [
        final_cv_scores["test_roc_auc"].std(ddof=1),
        final_cv_scores["test_f1"].std(ddof=1),
        final_cv_scores["test_precision"].std(ddof=1),
        final_cv_scores["test_recall"].std(ddof=1),
        final_cv_scores["test_accuracy"].std(ddof=1)
    ]
})

final_stability_metrics_df["Mean ± Std"] = (
    final_stability_metrics_df["CV Mean"].map(lambda x: f"{x:.4f}") +
    " ± " +
    final_stability_metrics_df["CV Std Dev"].map(lambda x: f"{x:.4f}")
)

print("\n=== Final Model Stability Metrics (Cross-validation Mean ± Std Deviation) ===")
print(final_stability_metrics_df[["Metric", "Mean ± Std"]].to_string(index=False))

"""Not enough resources lol...

### Ezra ー Mechanical Failure Analysis

Notes:
- LightGBM is a gradient boosting model that:
  - Handles complex feature interactions well
  - Trains faster than traditional Random Forests
  - Often achieves better performance on tabular data
  - Uses leaf-wise tree growth for optimal splits

- LightGBM typically achieves excellent scores on benchmark datasets and has been demonstrated to outperform tree-based models like Random Forest

- This implementation uses scale_pos_weight for cost-sensitive learning to handle the class imbalance naturally without synthetic data generation

#### Retrain Champion (LightGBM Classifier) Model using best ablation parameters
"""

lgbm_final = LGBMClassifier(
    n_estimators=final_model_params.get("n_estimators", 100),
    num_leaves=final_model_params.get("num_leaves", 31),
    max_depth=final_model_params.get("max_depth", 7),
    learning_rate=final_model_params.get("learning_rate", 0.1),
    subsample=final_model_params.get("subsample", 0.8),
    colsample_bytree=final_model_params.get("colsample_bytree", 0.8),
    scale_pos_weight=scale_pos_weight,  # ← IMPROVED: Handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

pipeline_final = ImbPipeline(
    steps=core_preprocessing_pipeline.steps + [("classifier", lgbm_final)]
)

pipeline_final.fit(X_train, y_train)

y_prob = pipeline_final.predict_proba(X_test)[:,1]

# IMPROVED: Compute optimal threshold using Precision-Recall curve
# This addresses the trade-off discovered in the model evaluation (Precision vs Recall)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate F1-scores for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print("\n=== OPTIMIZED THRESHOLD ANALYSIS ===")
print(f"Optimal Threshold (maximizing F1-score): {optimal_threshold:.3f}")
print(f"F1-score at optimal threshold: {f1_scores[best_idx]:.4f}")

# Show metrics at different thresholds
print("\n=== Threshold Search Results ===")
threshold_analysis = []
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_th = (y_prob >= th).astype(int)
    fp = ((y_test == 0) & (y_pred_th == 1)).sum()
    fn = ((y_test == 1) & (y_pred_th == 0)).sum()
    tn = ((y_test == 0) & (y_pred_th == 0)).sum()
    tp = ((y_test == 1) & (y_pred_th == 1)).sum()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    threshold_analysis.append({
        "Threshold": th,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

threshold_df = pd.DataFrame({
    "Threshold": [r["Threshold"] for r in threshold_analysis],
    "True Positives": [r["True Positives"] for r in threshold_analysis],
    "False Positives": [r["False Positives"] for r in threshold_analysis],
    "True Negatives": [r["True Negatives"] for r in threshold_analysis],
    "False Negatives": [r["False Negatives"] for r in threshold_analysis],
    "Accuracy": [f"{r['Accuracy']:.4f}" for r in threshold_analysis],
    "Precision": [f"{r['Precision']:.4f}" for r in threshold_analysis],
    "Recall": [f"{r['Recall']:.4f}" for r in threshold_analysis],
    "F1-Score": [f"{r['F1-Score']:.4f}" for r in threshold_analysis]
})

# Apply optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

# Option B: business goal = catch defaults (high recall)
TARGET_RECALL = 0.70  # change this to whatever recall target the business requires
candidates = [r for r in threshold_analysis if r["Recall"] >= TARGET_RECALL]
if candidates:
    # Choose the threshold that maximizes precision (minimizes false positives) while meeting recall target
    best_target = max(candidates, key=lambda r: r["Precision"])
    recall_target_threshold = best_target["Threshold"]
    recall_target_precision = best_target["Precision"]
    recall_target_recall = best_target["Recall"]
else:
    recall_target_threshold = optimal_threshold
    recall_target_precision = None
    recall_target_recall = None

print(f"\n=== Business Target Recall Analysis (target={TARGET_RECALL:.0%}) ===")
if candidates:
    print(f"Chosen threshold: {recall_target_threshold:.3f} (precision={recall_target_precision:.4f}, recall={recall_target_recall:.4f})")
else:
    print("No threshold in the pre-defined search list reached the desired recall; using F1-optimal threshold instead.")

# Select final predictions based on business recall target
y_pred = (y_prob >= recall_target_threshold).astype(int)

results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "probability_default": y_prob
})

print("\n=== Predictions at Chosen Threshold ===")
print(results_df)

# Confusion matrix at chosen threshold
from sklearn.metrics import confusion_matrix as cm
conf_mat = cm(y_test, y_pred)
print("\n=== Confusion Matrix at Chosen Threshold ===")
print(f"True Negatives: {conf_mat[0,0]}")
print(f"False Positives: {conf_mat[0,1]}")
print(f"False Negatives: {conf_mat[1,0]}")
print(f"True Positives: {conf_mat[1,1]}")

false_pos = (y_test == 0) & (y_pred == 1)
false_neg = (y_test == 1) & (y_pred == 0)

fp_cases = X_test[false_pos]
fn_cases = X_test[false_neg]

print("\n=== False Positive Cases (model predicts default but actually paid) ===")
print(fp_cases)

print("\n=== False Negative Cases (model predicts paid but actually defaulted) ===")
print(fn_cases)
