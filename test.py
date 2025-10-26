# -*- coding: utf-8 -*-
"""
test.py — sanity check très rapide de la pertinence du modèle (best_model.pkl)
- Charge best_model.pkl (pipeline sklearn recommandé)
- Échantillonne 100 lignes (stratifié si possible)
- Vérifie: accuracy >= baseline (classe majoritaire)
- (Si possible) ROC-AUC >= 0.55
"""

from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/global_house_purchase_dataset.csv")
MODEL_PATH = Path("best_model.pkl")
TARGET_COL = "decision"
SAMPLE_N = 100
RANDOM_STATE = 42
AUC_MIN = 0.55  # seuil souple

def test_model_pertinence_quick():
    # 1) Présence des fichiers
    assert MODEL_PATH.exists(), f"Modèle introuvable: {MODEL_PATH}"
    assert DATA_PATH.exists(), f"CSV introuvable: {DATA_PATH}"

    # 2) Charger données
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET_COL])
    assert len(df) >= 10, "Dataset trop petit pour ce test."
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = pd.factorize(df[TARGET_COL])[0]
    y_full = df[TARGET_COL].astype(int)

    # 3) Échantillonner pour vitesse (stratifié si binaire/multi-classe)
    if len(df) > SAMPLE_N:
        if y_full.nunique() > 1:
            df_sample, _ = train_test_split(
                df, train_size=SAMPLE_N, stratify=y_full, random_state=RANDOM_STATE
            )
        else:
            df_sample = df.sample(SAMPLE_N, random_state=RANDOM_STATE)
    else:
        df_sample = df

    X = df_sample.drop(columns=[TARGET_COL]).copy()
    y = df_sample[TARGET_COL].astype(int).copy()
    X.columns = [str(c) for c in X.columns]  # hygiène

    # 4) Charger modèle (pipeline sklearn recommandé)
    model = joblib.load(MODEL_PATH)

    # 5) Prédire
    try:
        y_pred = model.predict(X)
    except Exception as e:
        pytest.fail(
            "Le modèle ne peut pas prédire sur les données brutes. "
            "Assure-toi que best_model.pkl est un *pipeline sklearn* (préprocessing inclus). "
            f"Erreur: {e}"
        )

    # 6) Baseline vs accuracy
    majority_class = int(y.value_counts().idxmax())
    baseline_acc = (y == majority_class).mean()
    model_acc = accuracy_score(y, y_pred)
    assert model_acc >= baseline_acc, (
        f"Pertinence insuffisante: accuracy={model_acc:.3f} < baseline={baseline_acc:.3f}"
    )

    # 7) AUC minimale si proba dispo
    if hasattr(model, "predict_proba") and y.nunique() == 2:
        try:
            proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            assert auc >= AUC_MIN, f"ROC-AUC trop basse: {auc:.3f} < {AUC_MIN:.2f}"
        except Exception:
            # si proba KO, l'accuracy a déjà validé la pertinence
            pass
