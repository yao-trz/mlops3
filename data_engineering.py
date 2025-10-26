# -*- coding: utf-8 -*-
"""
train_and_log_models.py
- EDA minimale + préprocessing robuste (imputation, OHE, scaling)
- Entraîne 3 modèles: LogisticRegression, DecisionTree, RandomForest
- Log des runs dans MLflow (métriques + artefacts)
- Sauvegarde localement le meilleur pipeline: best_model.pkl
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from mlflow.models.signature import infer_signature

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "global_house_purchase_dataset.csv")
TARGET_COL = "decision"
EXPERIMENT_NAME = "HousePurchase_Classification"

def build_preprocessor(df: pd.DataFrame):
    # Sépare num/cat automatiquement
    X = df.drop(columns=[TARGET_COL])
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

def evaluate_and_log(model_name, pipeline, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Probas si dispo
        has_proba = hasattr(pipeline, "predict_proba")
        y_proba = pipeline.predict_proba(X_test)[:, 1] if has_proba else None

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        })

        # Log params (y compris ceux du classifieur)
        clf = pipeline.named_steps["clf"]
        mlflow.log_params(clf.get_params())

        # Signature MLflow
        X_example = X_train[:5].copy()
        signature = infer_signature(X_example, pipeline.predict(X_example))

        # Log du pipeline complet
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_example.iloc[[0]]
        )

        # Log text report & confusion matrix (en tant qu'artifacts)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        os.makedirs("artifacts", exist_ok=True)
        with open(f"artifacts/{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        np.savetxt(f"artifacts/{model_name}_confusion_matrix.txt", cm, fmt="%d")

        mlflow.log_artifacts("artifacts")

        return {"name": model_name, "f1": f1, "pipeline": pipeline}

def main():
    # Chargement
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    # On impute dans le pipeline, donc on garde les NaN ici (optionnel)
    # Si tu préfères dropper les lignes 100% vides de target :
    df = df.dropna(subset=[TARGET_COL])

    y = df[TARGET_COL].astype("category").cat.codes if df[TARGET_COL].dtype == "object" else df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # Modèles
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=14, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", clf)
        ])
        res = evaluate_and_log(name, pipe, X_train, X_test, y_train, y_test)
        results.append(res)

    # Choix du meilleur au F1
    best = max(results, key=lambda r: r["f1"])
    joblib.dump(best["pipeline"], "best_model.pkl")

if __name__ == "__main__":
    main()
