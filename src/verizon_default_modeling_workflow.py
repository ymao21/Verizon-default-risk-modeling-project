"""Verizon device-financing default risk modeling workflow.

This script is a reusable implementation scaffold for the modeling approach
described in the project reports. It expects applicant-level data with a binary
default target and features such as FICO score, device cost, down payment,
borrower age, payment type, and application date.

The raw project data is not included in this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional.
    XGBClassifier = None


TARGET = "del90"
NUMERIC_FEATURES = ["fico", "device_cost", "down_payment", "borrower_age"]
CATEGORICAL_FEATURES = ["payment_type"]
DATE_COLUMN = "application_date"


@dataclass
class BusinessAssumptions:
    """Inputs for translating classification outcomes into business value."""

    lifetime_value_per_paying_customer: float = 150.0
    cost_per_default: float = 741.0


def load_applicant_data(path: str | Path) -> pd.DataFrame:
    """Load applicant-level device-financing data from CSV or Parquet."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create month/year features when an application date column is present."""

    df = df.copy()
    if DATE_COLUMN not in df.columns:
        return df

    parsed = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["application_year"] = parsed.dt.year
    df["application_month"] = parsed.dt.month
    return df


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical model inputs."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ]
    )


def candidate_models(random_state: int = 42) -> dict[str, object]:
    """Return model candidates used for comparison."""

    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )

    return models


def best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Choose the probability threshold that maximizes F1 score."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return 0.5
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index])


def evaluate_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute classification metrics at a selected decision threshold."""

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


def compare_models(
    df: pd.DataFrame,
    target: str = TARGET,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline], tuple[pd.DataFrame, pd.Series]]:
    """Train candidate models and return model-comparison metrics."""

    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    required = available_numeric + available_categorical + [target]
    model_df = df.dropna(subset=[target]).copy()

    if not available_numeric:
        raise ValueError("No expected numeric model features were found.")
    if target not in model_df.columns:
        raise ValueError(f"Target column not found: {target}")

    x = model_df[required].drop(columns=[target])
    y = model_df[target].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(available_numeric, available_categorical)
    rows = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_prob = pipeline.predict_proba(x_test)[:, 1]
        threshold = best_f1_threshold(y_test, y_prob)
        rows.append({"model": name, **evaluate_predictions(y_test, y_prob, threshold)})
        fitted[name] = pipeline

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False), fitted, (x_test, y_test)


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return coefficients from the fitted logistic regression pipeline."""

    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    coefficients = classifier.coef_.ravel()

    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )


def estimate_cash_flow(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    assumptions: BusinessAssumptions,
) -> dict[str, float]:
    """Estimate cash-flow impact from approval decisions.

    Applicants below the threshold are approved. Applicants at or above the
    threshold are flagged as high risk and declined or sent to additional review.
    """

    approve = y_prob < threshold
    actual_default = y_true.astype(int).to_numpy() == 1

    paying_approved = int((approve & ~actual_default).sum())
    risky_approved = int((approve & actual_default).sum())
    risky_declined = int((~approve & actual_default).sum())
    safe_declined = int((~approve & ~actual_default).sum())

    profit = paying_approved * assumptions.lifetime_value_per_paying_customer
    default_loss = risky_approved * assumptions.cost_per_default
    missed_profit = safe_declined * assumptions.lifetime_value_per_paying_customer
    avoided_loss = risky_declined * assumptions.cost_per_default
    net_cash_flow = profit + avoided_loss - default_loss - missed_profit

    return {
        "approved_paying_customers": paying_approved,
        "approved_default_customers": risky_approved,
        "declined_default_customers": risky_declined,
        "declined_paying_customers": safe_declined,
        "profit_from_paying_customers": profit,
        "loss_from_defaults": default_loss,
        "avoided_default_loss": avoided_loss,
        "missed_profit": missed_profit,
        "net_cash_flow": net_cash_flow,
        "cash_flow_per_applicant": net_cash_flow / len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to applicant CSV or Parquet file.")
    parser.add_argument("--target", default=TARGET, help="Binary default target column.")
    args = parser.parse_args()

    df = add_temporal_features(load_applicant_data(args.data))
    metrics, fitted_models, test_data = compare_models(df, target=args.target)

    print("\nModel comparison")
    print(metrics.to_string(index=False))

    if "Logistic Regression" in fitted_models:
        print("\nLogistic regression coefficient hierarchy")
        print(logistic_coefficients(fitted_models["Logistic Regression"]).head(15).to_string(index=False))

        x_test, y_test = test_data
        y_prob = fitted_models["Logistic Regression"].predict_proba(x_test)[:, 1]
        threshold = float(metrics.loc[metrics["model"] == "Logistic Regression", "threshold"].iloc[0])
        impact = estimate_cash_flow(y_test, y_prob, threshold, BusinessAssumptions())
        print("\nCash-flow impact estimate")
        print(pd.Series(impact).to_string())


if __name__ == "__main__":
    main()
