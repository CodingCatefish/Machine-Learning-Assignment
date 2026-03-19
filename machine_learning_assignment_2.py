from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import environment dependent
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = exc


RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_COMPONENTS = 25
HIGH_CARDINALITY_THRESHOLD = 50
SCORING = ["roc_auc", "f1", "precision", "recall", "accuracy"]
CLASS_LABELS = ["Low Likelihood to Default", "High Likelihood to Default"]
TARGET_COLUMN = "target"
LOCAL_DATA_PATH = Path(__file__).with_name("loan.csv")

DROP_COLUMNS = [
    TARGET_COLUMN,
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

PERCENTAGE_COLUMNS = ("int_rate", "revol_util")
NUMERIC_RESCUE_COLUMNS = (
    "annual_inc_joint",
    "dti_joint",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_il_6m",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
)

MODEL_BASELINES: dict[str, dict[str, Any]] = {
    "LightGBM": {
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "n_jobs": -1,
        "verbose": -1,
    },
    "BernoulliNaiveBayes": {
        "alpha": 1.0,
        "binarize": 0.0,
        "fit_prior": True,
    },
    "LogisticRegression": {
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
    },
}

MODEL_ABLATIONS: dict[str, list[dict[str, Any]]] = {
    "LightGBM": [
        {
            "Experiment": "E1_num_leaves_63",
            "Changed_Parameter": "num_leaves",
            "Old_Value": 31,
            "New_Value": 63,
            "Params": {"num_leaves": 63},
        },
        {
            "Experiment": "E2_learning_rate_003",
            "Changed_Parameter": "learning_rate",
            "Old_Value": 0.05,
            "New_Value": 0.03,
            "Params": {"learning_rate": 0.03},
        },
        {
            "Experiment": "E3_n_estimators_500",
            "Changed_Parameter": "n_estimators",
            "Old_Value": 300,
            "New_Value": 500,
            "Params": {"n_estimators": 500},
        },
        {
            "Experiment": "E4_min_child_samples_10",
            "Changed_Parameter": "min_child_samples",
            "Old_Value": 20,
            "New_Value": 10,
            "Params": {"min_child_samples": 10},
        },
    ],
    "BernoulliNaiveBayes": [
        {
            "Experiment": "E1_alpha_05",
            "Changed_Parameter": "alpha",
            "Old_Value": 1.0,
            "New_Value": 0.5,
            "Params": {"alpha": 0.5},
        },
        {
            "Experiment": "E2_alpha_20",
            "Changed_Parameter": "alpha",
            "Old_Value": 1.0,
            "New_Value": 2.0,
            "Params": {"alpha": 2.0},
        },
        {
            "Experiment": "E3_fit_prior_false",
            "Changed_Parameter": "fit_prior",
            "Old_Value": True,
            "New_Value": False,
            "Params": {"fit_prior": False},
        },
        {
            "Experiment": "E4_binarize_01",
            "Changed_Parameter": "binarize",
            "Old_Value": 0.0,
            "New_Value": 0.1,
            "Params": {"binarize": 0.1},
        },
    ],
    "LogisticRegression": [
        {
            "Experiment": "E1_C_05",
            "Changed_Parameter": "C",
            "Old_Value": 1.0,
            "New_Value": 0.5,
            "Params": {"C": 0.5},
        },
        {
            "Experiment": "E2_C_20",
            "Changed_Parameter": "C",
            "Old_Value": 1.0,
            "New_Value": 2.0,
            "Params": {"C": 2.0},
        },
        {
            "Experiment": "E3_class_weight_balanced",
            "Changed_Parameter": "class_weight",
            "Old_Value": None,
            "New_Value": "balanced",
            "Params": {"class_weight": "balanced"},
        },
        {
            "Experiment": "E4_solver_liblinear",
            "Changed_Parameter": "solver",
            "Old_Value": "lbfgs",
            "New_Value": "liblinear",
            "Params": {"solver": "liblinear"},
        },
    ],
}


def encode_sub_grade(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    grade_text = str(value).strip().upper()
    if len(grade_text) < 2:
        return np.nan

    grade_score = {"A": 7.0, "B": 6.0, "C": 5.0, "D": 4.0, "E": 3.0, "F": 2.0, "G": 1.0}
    suffix_score = {"1": 0.8, "2": 0.6, "3": 0.4, "4": 0.2, "5": 0.0}

    return grade_score.get(grade_text[0], np.nan) + suffix_score.get(grade_text[-1], 0.0)


class NotebookAlignedEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        drop_columns: list[str] | None = None,
        high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
    ) -> None:
        self.drop_columns = drop_columns or []
        self.high_cardinality_threshold = high_cardinality_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NotebookAlignedEncoder":
        frame = self._coerce_frame(X)
        self.kept_columns_ = [column for column in frame.columns if column not in self.drop_columns]
        filtered = frame.reindex(columns=self.kept_columns_).copy()

        object_columns = filtered.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.high_cardinality_columns_ = sorted(
            column
            for column in object_columns
            if filtered[column].nunique(dropna=True) > self.high_cardinality_threshold
        )

        prepared = self._prepare(filtered)
        self.feature_names_ = prepared.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["kept_columns_", "high_cardinality_columns_", "feature_names_"])
        frame = self._coerce_frame(X).reindex(columns=self.kept_columns_).copy()
        prepared = self._prepare(frame)
        aligned = prepared.reindex(columns=self.feature_names_, fill_value=0.0)
        return aligned.astype(np.float32)

    @staticmethod
    def _coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X).copy()

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.drop(columns=self.high_cardinality_columns_, errors="ignore").copy()

        for column in PERCENTAGE_COLUMNS:
            if column in working.columns:
                working[column] = (
                    working[column]
                    .astype(str)
                    .str.rstrip("%")
                    .replace({"nan": np.nan, "None": np.nan})
                )
                working[column] = pd.to_numeric(working[column], errors="coerce")

        for column in NUMERIC_RESCUE_COLUMNS:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")

        if "sub_grade" in working.columns:
            working["sub_grade"] = working["sub_grade"].apply(encode_sub_grade)

        if "term" in working.columns:
            working["term"] = working["term"].map({"36 months": 0, "60 months": 1})

        if "initial_list_status" in working.columns:
            working["initial_list_status"] = working["initial_list_status"].map({"f": 0, "w": 1})

        if "verification_status" in working.columns:
            working["verification_status"] = working["verification_status"].map(
                {"Not Verified": 0, "Verified": 1, "Source Verified": 1}
            )

        categorical_columns = working.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if categorical_columns:
            working = pd.get_dummies(working, columns=categorical_columns, dtype=float)

        working = working.apply(pd.to_numeric, errors="coerce")
        return working


class SafePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = PCA_COMPONENTS, random_state: int = RANDOM_STATE) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: pd.Series | None = None) -> "SafePCA":
        n_components = max(1, min(self.n_components, X.shape[0], X.shape[1]))
        self.n_components_ = n_components
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["pca_", "n_components_"])
        return self.pca_.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        check_is_fitted(self, ["pca_"])
        return self.pca_.explained_variance_ratio_


class AutoDeviceLGBMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, **lgbm_params: Any) -> None:
        self.lgbm_params = dict(lgbm_params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self.lgbm_params)

    def set_params(self, **params: Any) -> "AutoDeviceLGBMClassifier":
        self.lgbm_params.update(params)
        return self

    def fit(self, X: np.ndarray, y: pd.Series, **fit_params: Any) -> "AutoDeviceLGBMClassifier":
        if LGBMClassifier is None:
            raise ImportError(
                "LightGBM is required for this assignment. Install it in your environment before running the script."
            ) from LIGHTGBM_IMPORT_ERROR

        base_params = dict(self.lgbm_params)
        device_attempts = (
            ("gpu", {**base_params, "device_type": "gpu"}),
            ("cpu", {**base_params, "device_type": "cpu"}),
        )
        last_error: Exception | None = None

        for device_name, device_params in device_attempts:
            try:
                estimator = LGBMClassifier(**device_params)
                estimator.fit(X, y, **fit_params)
                self.estimator_ = estimator
                self.fit_device_ = device_name
                self.classes_ = estimator.classes_
                if device_name == "cpu" and last_error is not None:
                    warnings.warn(
                        f"LightGBM GPU training failed; falling back to CPU. Original error: {last_error}",
                        RuntimeWarning,
                    )
                return self
            except Exception as exc:  # pragma: no cover - depends on local GPU/OpenCL setup
                last_error = exc

        raise RuntimeError("LightGBM could not fit on GPU or CPU.") from last_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict_proba(X)


def load_credit_risk_data(data_path: Path = LOCAL_DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df[TARGET_COLUMN] = np.where(df["loan_status"].eq("Fully Paid"), 0, 1)
    return df


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "notebook_encoder",
                NotebookAlignedEncoder(
                    drop_columns=DROP_COLUMNS,
                    high_cardinality_threshold=HIGH_CARDINALITY_THRESHOLD,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", MinMaxScaler()),
            ("pca", SafePCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ]
    )


def build_estimator(model_name: str, overrides: dict[str, Any] | None = None) -> BaseEstimator:
    params = {**MODEL_BASELINES[model_name], **(overrides or {})}

    if model_name == "LightGBM":
        return AutoDeviceLGBMClassifier(**params)
    if model_name == "BernoulliNaiveBayes":
        return BernoulliNB(**params)
    if model_name == "LogisticRegression":
        return LogisticRegression(**params)

    raise ValueError(f"Unsupported model name: {model_name}")


def build_model_pipeline(model_name: str, overrides: dict[str, Any] | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", build_estimator(model_name, overrides)),
        ]
    )


def build_cv_row(
    experiment_name: str,
    changed_parameter: str,
    old_value: Any,
    new_value: Any,
    cv_scores: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "Experiment": experiment_name,
        "Changed_Parameter": changed_parameter,
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
        "Runtime_sec": elapsed_seconds,
    }


def run_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    print("Part B: Model Selection")

    results: list[dict[str, Any]] = []
    for model_name in MODEL_BASELINES:
        start_time = time.time()
        print(f"\nStarting {model_name}...")
        pipeline = build_model_pipeline(model_name)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        elapsed_seconds = time.time() - start_time
        results.append(
            {
                "Model": model_name,
                "ROC_AUC": cv_scores["test_roc_auc"].mean(),
                "F1": cv_scores["test_f1"].mean(),
                "Precision": cv_scores["test_precision"].mean(),
                "Recall": cv_scores["test_recall"].mean(),
                "Accuracy": cv_scores["test_accuracy"].mean(),
                "Runtime_sec": elapsed_seconds,
            }
        )
        print(f"{model_name} finished in {elapsed_seconds:.2f} seconds")

    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    print("\n=== Model Selection Results ===")
    print(results_df.to_string(index=False))
    return results_df


def run_ablation_study(
    champion_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    print(f"\nPart C: Ablation Testing for {champion_name}")

    baseline_params = MODEL_BASELINES[champion_name]
    ablation_rows: list[dict[str, Any]] = []

    baseline_start = time.time()
    baseline_pipeline = build_model_pipeline(champion_name, baseline_params)
    baseline_scores = cross_validate(
        baseline_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    baseline_elapsed = time.time() - baseline_start
    baseline_row = build_cv_row("Baseline", "None", "-", "-", baseline_scores, baseline_elapsed)
    ablation_rows.append(baseline_row)

    for experiment in MODEL_ABLATIONS[champion_name]:
        print(f"Running {experiment['Experiment']}...")
        experiment_start = time.time()
        experiment_params = {**baseline_params, **experiment["Params"]}
        experiment_pipeline = build_model_pipeline(champion_name, experiment_params)
        experiment_scores = cross_validate(
            experiment_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        experiment_elapsed = time.time() - experiment_start
        ablation_rows.append(
            build_cv_row(
                experiment_name=experiment["Experiment"],
                changed_parameter=experiment["Changed_Parameter"],
                old_value=experiment["Old_Value"],
                new_value=experiment["New_Value"],
                cv_scores=experiment_scores,
                elapsed_seconds=experiment_elapsed,
            )
        )

    ablation_log_df = pd.DataFrame(ablation_rows)
    for metric in ("ROC_AUC", "F1", "Precision", "Recall", "Accuracy"):
        ablation_log_df[f"Delta_{metric}"] = ablation_log_df[metric] - baseline_row[metric]

    ablation_log_df = ablation_log_df.sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    best_experiment_name = ablation_log_df.loc[0, "Experiment"]
    best_params = dict(baseline_params)

    if best_experiment_name != "Baseline":
        winning_experiment = next(
            experiment
            for experiment in MODEL_ABLATIONS[champion_name]
            if experiment["Experiment"] == best_experiment_name
        )
        best_params.update(winning_experiment["Params"])

    print("\n=== Ablation Log ===")
    print(ablation_log_df.to_string(index=False))
    print(f"\nSelected champion configuration: {best_experiment_name}")
    print(f"Final {champion_name} parameters: {best_params}")

    return ablation_log_df, best_params, best_experiment_name


def evaluate_final_model(
    champion_name: str,
    champion_params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Pipeline:
    print("\nPart D: Final Champion Evaluation")
    final_pipeline = build_model_pipeline(champion_name, champion_params)
    final_pipeline.fit(X_train, y_train)

    classifier = final_pipeline.named_steps["classifier"]
    if hasattr(classifier, "fit_device_"):
        print(f"{champion_name} trained using device: {classifier.fit_device_}")

    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "ROC_AUC",
                "Score": roc_auc_score(y_test, y_prob),
            },
            {
                "Metric": "F1",
                "Score": f1_score(y_test, y_pred),
            },
            {
                "Metric": "Precision",
                "Score": precision_score(y_test, y_pred),
            },
            {
                "Metric": "Recall",
                "Score": recall_score(y_test, y_pred),
            },
            {
                "Metric": "Accuracy",
                "Score": accuracy_score(y_test, y_pred),
            },
        ]
    )

    print("\n=== Final Test Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    plot_confusion_heatmap(y_test, y_pred)
    return final_pipeline


def plot_confusion_heatmap(y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title("Final Champion Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Part A: Data Prep & Pipeline Engineering")
    df = load_credit_risk_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Loaded filtered dataset with shape: {df.shape}")
    print(f"Local dataset path: {LOCAL_DATA_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results_df = run_model_selection(X_train, y_train, cv)
    champion_name = results_df.loc[0, "Model"]
    print(f"\nChampion selected by ROC_AUC: {champion_name}")

    _, champion_params, _ = run_ablation_study(champion_name, X_train, y_train, cv)
    evaluate_final_model(champion_name, champion_params, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
