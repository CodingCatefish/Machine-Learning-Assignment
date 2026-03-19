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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

# Model Selection Phase - Lionel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from tqdm import tqdm

# Ablations and tuning phase - Justin

# Mechanical failure analysis - Ezra

# Decision Making and consolidation - Jing Hai

df = pl.read_csv("loan.csv")

print(df)

"""###Yixuan - Data Preperation"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

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

# Keep imputed column positions stable so SMOTENC can safely reference categorical columns by index.
new_cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

# 6. Pipeline Step 2: Resampling
# Downsample the majority class first, then let SMOTENC safely rebalance.
undersampler = RandomUnderSampler(
    sampling_strategy={0: 80000}, # Shrink "Fully Paid" to 80,000
    random_state=42
)

smote_nc = SMOTENC(
    categorical_features=new_cat_indices,
    random_state=42,
    sampling_strategy='auto' # Raise the minority class to match the undersampled majority class
)

# 7. Pipeline Step 3: Scaling & Encoding
encoder_scaler_step = ColumnTransformer(
    transformers=[
        ('num_scaler', RobustScaler(), list(range(len(num_cols)))),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), new_cat_indices)
    ],
    remainder='passthrough'
)

# 8. Assemble the Core Preprocessing Pipeline
core_preprocessing_pipeline = ImbPipeline(steps=[
    ('imputation', imputer_step),
    ('undersampling', undersampler),
    ('smote_nc', smote_nc),
    ('scaling_encoding', encoder_scaler_step)
])

print("Data Prep and Pipeline Engineering is complete. Ready for handoff.")

# This section is used to generate visuals for Data Preparation and Pipeline Engineering phase.
print("--- Generating Visuals for Data Preparation and Pipeline Engineering phase ---")




'''
# Generating a "Before and After" diagram, showing the sheer volume of the original training data side-by-side with the artificially balanced data that the pipeline has created.
# Show :
# 1. Extreme imbalance.
# 2. RandomUnderSampler chopped the massive 'Fully Paid' class down to 80,000.
# 3. SMOTE-NC synthetically raises the minority class to match it, creating a balanced dataset for the models to learn from.

# BEFORE: Original Training Data Distribution
before_counts = y_train.value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# MIDDLE: After Imputation & Undersampling
# Clone the preprocessing objects so the visualization does not mutate the training pipeline.
imputer_viz = clone(imputer_step)
undersampler_viz = clone(undersampler)
smote_viz = clone(smote_nc)

X_train_imputed = imputer_viz.fit_transform(X_train)
X_train_under, y_train_under = undersampler_viz.fit_resample(X_train_imputed, y_train)
middle_counts = pd.Series(y_train_under).value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# AFTER: After SMOTE-NC Oversampling
X_train_smote, y_train_smote = smote_viz.fit_resample(X_train_under, y_train_under)
after_counts = pd.Series(y_train_smote).value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# --- Plotting the Diagram for the Slide ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Plot 1: Original Imbalance
sns.barplot(x=before_counts.index, y=before_counts.values, ax=axes[0], palette=["#1f77b4", "#ff7f0e"])
axes[0].set_title(f"1. Original Data\nTotal Rows: {len(y_train):,}")
axes[0].bar_label(axes[0].containers[0], fmt='%d')

# Plot 2: After Undersampling
sns.barplot(x=middle_counts.index, y=middle_counts.values, ax=axes[1], palette=["#1f77b4", "#ff7f0e"])
axes[1].set_title(f"2. After Undersampling\nTotal Rows: {len(y_train_under):,}")
axes[1].bar_label(axes[1].containers[0], fmt='%d')

# Plot 3: After SMOTE-NC
sns.barplot(x=after_counts.index, y=after_counts.values, ax=axes[2], palette=["#1f77b4", "#ff7f0e"])
axes[2].set_title(f"3. After SMOTE-NC\nTotal Rows: {len(y_train_smote):,}")
axes[2].bar_label(axes[2].containers[0], fmt='%d')

plt.tight_layout()
plt.show()
'''


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# This section is used to generate visuals for Data Preparation and Pipeline Engineering phase.
print("--- Generating Visuals for Data Preparation and Pipeline Engineering phase ---")

# BEFORE: Original Training Data Distribution
before_counts = y_train.value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# MIDDLE: After Imputation & Undersampling
# Clone the preprocessing objects so the visualization does not mutate the training pipeline.
imputer_viz = clone(imputer_step)
undersampler_viz = clone(undersampler)
smote_viz = clone(smote_nc)

X_train_imputed = imputer_viz.fit_transform(X_train)
X_train_under, y_train_under = undersampler_viz.fit_resample(X_train_imputed, y_train)
middle_counts = pd.Series(y_train_under).value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# AFTER: After SMOTE-NC Oversampling
X_train_smote, y_train_smote = smote_viz.fit_resample(X_train_under, y_train_under)
after_counts = pd.Series(y_train_smote).value_counts().rename(index={0: "Fully Paid (0)", 1: "Default (1)"})

# --- Plotting the Diagram for the Slide ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Plot 1: Original Imbalance
sns.barplot(x=before_counts.index, y=before_counts.values, ax=axes[0], hue=before_counts.index, palette=["#1f77b4", "#ff7f0e"], legend=False)
axes[0].set_title(f"1. Original Data\nTotal Rows: {len(y_train):,}")
axes[0].bar_label(axes[0].containers[0], fmt='%d')

# Plot 2: After Undersampling (Reduced Fully Paid to 80k)
sns.barplot(x=middle_counts.index, y=middle_counts.values, ax=axes[1], hue=middle_counts.index, palette=["#1f77b4", "#ff7f0e"], legend=False)
axes[1].set_title(f"2. After Undersampling\nTotal Rows: {len(y_train_under):,}")
axes[1].bar_label(axes[1].containers[0], fmt='%d')

# Plot 3: After SMOTE-NC (Balanced Classes)
sns.barplot(x=after_counts.index, y=after_counts.values, ax=axes[2], hue=after_counts.index, palette=["#1f77b4", "#ff7f0e"], legend=False)
axes[2].set_title(f"3. After SMOTE-NC (Balanced)\nTotal Rows: {len(y_train_smote):,}")
axes[2].bar_label(axes[2].containers[0], fmt='%d')

plt.tight_layout()
plt.show()

"""###Lionel - Model Selection"""

print("Part B: Model Selection")

# 1. Models: Random Forest, Bernoulli Naive Base, SVM

models = {
     "RandomForest": RandomForestClassifier(
         n_estimators=100,
         max_depth=None,
         n_jobs= 1,
         random_state=42
     ),

    "BernoulliNaiveBayes": BernoulliNB(),

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
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

# Set up selected baseline champion (Random Forest)
rf_baseline_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": 1,
    "random_state": 42
}

# Four experiments; each changes only one parameter from the baseline
ablation_experiments = [
    {
        "Experiment": "E1_n_estimators_200",
        "Changed_Parameter": "n_estimators",
        "Old_Value": 100,
        "New_Value": 200,
        "Params": {**rf_baseline_params, "n_estimators": 200}
    },
    {
        "Experiment": "E2_max_depth_20",
        "Changed_Parameter": "max_depth",
        "Old_Value": None,
        "New_Value": 20,
        "Params": {**rf_baseline_params, "max_depth": 20}
    },
    {
        "Experiment": "E3_min_samples_split_5",
        "Changed_Parameter": "min_samples_split",
        "Old_Value": 2,
        "New_Value": 5,
        "Params": {**rf_baseline_params, "min_samples_split": 5}
    },
    {
        "Experiment": "E4_min_samples_leaf_2",
        "Changed_Parameter": "min_samples_leaf",
        "Old_Value": 1,
        "New_Value": 2,
        "Params": {**rf_baseline_params, "min_samples_leaf": 2}
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
print("\nRunning baseline Random Forest...")
baseline_start = time.time()

baseline_model = RandomForestClassifier(**rf_baseline_params)
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

    model = RandomForestClassifier(**exp["Params"])
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

print("\n=== Random Forest Ablation Log ===")
print(ablation_log_df.to_string(index=False))

print("\n=== Ranked by ROC_AUC ===")
print(ablation_sorted_df.to_string(index=False))

# Parameter selection.
best_row = ablation_sorted_df.iloc[0]
best_experiment_name = best_row["Experiment"]

if best_experiment_name == "Baseline":
    final_model_params = rf_baseline_params
else:
    final_model_params = next(
        exp["Params"] for exp in ablation_experiments
        if exp["Experiment"] == best_experiment_name
    )

print(f"\nFinal selected model from ablation: {best_experiment_name}")
print(f"Final model parameters: {final_model_params}")

# Cross validation with selected parameter.
final_model = RandomForestClassifier(**final_model_params)
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
- Random Forest usually wins in tabular datasets as it:
  - Handles complex relationships well
  - Works well with many features
  - Is less sensitive to noise

- This has been demonstrated in the Cross Validation stage, where it achieved the highest scores out of the 3 models in 3/5 of the classification model evaluation metrics. Those metrics being ROC_AUC, Precision and Accuracy.

- Ablation further showed that changing the min_samples_leaf had the greatest impact on the accuracy of the model out of all the other hyperparameters.

#### Retrain Champion (Random Forest Classifier) Model using 'E4_min_samples_leaf_2' parameters
"""

rf_e4 = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,   # E4 experiment change
    random_state=42,
    n_jobs=-1
)

pipeline_e4 = ImbPipeline(
    steps=core_preprocessing_pipeline.steps + [("classifier", rf_e4)]
)

pipeline_e4.fit(X_train, y_train)

y_pred = pipeline_e4.predict(X_test)
y_prob = pipeline_e4.predict_proba(X_test)[:,1]

results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "probability_default": y_prob
})

results_df

false_pos = (y_test == 0) & (y_pred == 1)
false_neg = (y_test == 1) & (y_pred == 0)

fp_cases = X_test[false_pos]
fn_cases = X_test[false_neg]

fp_cases

fn_cases
