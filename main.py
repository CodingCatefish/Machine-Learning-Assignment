# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COL = "target"
PRIMARY_METRIC = "ROC_AUC"


def load_dataset() -> pl.DataFrame:
    candidate_paths = [Path("loan/loan.csv"), Path("loan.csv")]

    for path in candidate_paths:
        if path.exists():
            print(f"Loading dataset from: {path}")
            return pl.read_csv(path)

    searched_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Dataset not found. Checked: {searched_paths}")


def prepare_target(df: pl.DataFrame) -> pd.DataFrame:
    filtered_df = df.filter(
        pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"])
    ).with_columns(
        pl.when(pl.col("loan_status") == "Fully Paid")
        .then(0)
        .otherwise(1)
        .alias(TARGET_COL)
    )

    df_pd = filtered_df.to_pandas()

    if {"issue_d", "earliest_cr_line"}.issubset(df_pd.columns):
        issue_dates = pd.to_datetime(df_pd["issue_d"], format="%b-%Y", errors="coerce")
        earliest_credit = pd.to_datetime(
            df_pd["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        df_pd["credit_hist_months"] = (issue_dates - earliest_credit).dt.days / 30

    for col in ["int_rate", "revol_util"]:
        if col in df_pd.columns and df_pd[col].dtype == "object":
            df_pd[col] = pd.to_numeric(
                df_pd[col].astype(str).str.rstrip("%"), errors="coerce"
            )

    numeric_rescue_cols = [
        "tot_cur_bal",
        "total_rev_hi_lim",
        "total_bal_il",
        "il_util",
        "max_bal_bc",
        "all_util",
    ]
    for col in numeric_rescue_cols:
        if col in df_pd.columns:
            df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

    return df_pd


def build_features(df_pd: pd.DataFrame):
    drop_cols = [
        TARGET_COL,
        "loan_status",
        "id",
        "member_id",
        "url",
        "desc",
        "emp_title",
        "title",
        "zip_code",
        "funded_amnt",
        "funded_amnt_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "out_prncp",
        "out_prncp_inv",
        "debt_settlement_flag",
        "hardship_flag",
        "pymnt_plan",
        "issue_d",
        "earliest_cr_line",
    ]

    cols_to_drop = [col for col in drop_cols if col in df_pd.columns]
    X = df_pd.drop(columns=cols_to_drop)
    y = df_pd[TARGET_COL]

    cat_cols_all = X.select_dtypes(include=["object", "string"]).columns.tolist()
    safe_cat_cols = [col for col in cat_cols_all if X[col].nunique() < 50]
    unsafe_cat_cols = [col for col in cat_cols_all if col not in safe_cat_cols]

    if unsafe_cat_cols:
        print(f"Dropping high-cardinality features: {unsafe_cat_cols}")
        X = X.drop(columns=unsafe_cat_cols)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in safe_cat_cols if col in X.columns]

    print(f"Using {len(num_cols)} numeric features and {len(cat_cols)} categorical features.")
    return X, y, num_cols, cat_cols


def build_preprocessing_pipeline(num_cols, cat_cols) -> ColumnTransformer:
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Missing"),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def build_models(y_train: pd.Series) -> dict:
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "BernoulliNaiveBayes": BernoulliNB(),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def evaluate_models(models, preprocessing, X_train, y_train) -> pd.DataFrame:
    scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(
            steps=[("preprocessing", preprocessing), ("classifier", model)]
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append(
            {
                "Model": name,
                "ROC_AUC": cv_scores["test_roc_auc"].mean(),
                "F1": cv_scores["test_f1"].mean(),
                "Precision": cv_scores["test_precision"].mean(),
                "Recall": cv_scores["test_recall"].mean(),
                "Accuracy": cv_scores["test_accuracy"].mean(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=PRIMARY_METRIC, ascending=False
    )
    return results_df


def fit_champion(preprocessing, model, X_train, y_train) -> Pipeline:
    pipeline = Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", model)]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def show_confusion_matrix(model_name, pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nConfusion matrix for champion model: {model_name}")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fully Paid (0)", "Default (1)"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("Part A: Data Prep & Pipeline Engineering")
    raw_df = load_dataset()
    df_pd = prepare_target(raw_df)
    X, y, num_cols, cat_cols = build_features(df_pd)

    print(f"Splitting data (input shape: {X.shape})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Original dataset retained. No undersampling or synthetic resampling applied.")
    preprocessing = build_preprocessing_pipeline(num_cols, cat_cols)

    print("\nPart B: Model Selection")
    models = build_models(y_train)
    results_df = evaluate_models(models, preprocessing, X_train, y_train)

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    champion_name = results_df.iloc[0]["Model"]
    champion_model = models[champion_name]
    print(f"\nChampion model based on {PRIMARY_METRIC}: {champion_name}")

    print("\nPart C: Holdout Evaluation")
    champion_pipeline = fit_champion(
        preprocessing, champion_model, X_train, y_train
    )
    show_confusion_matrix(champion_name, champion_pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
