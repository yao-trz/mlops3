import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILENAME_CANDIDATES = ["best_model.pkl", "model.pkl", "model (2).pkl"]
DATA_PATH = os.path.join("data", "global_house_purchase_dataset.csv")
TARGET_COL = "decision"

st.set_page_config(page_title="Prédiction achat immobilier", page_icon="🏠", layout="centered")
st.title("🏠 Prédiction de décision d'achat (pipeline sklearn)")

# -------- Chargement pipeline --------
def find_model_file():
    for name in MODEL_FILENAME_CANDIDATES:
        if os.path.exists(name):
            return name
    return None

model_path = find_model_file()
if not model_path:
    st.error(f"❌ Aucun modèle trouvé. Place un des fichiers suivants dans le répertoire courant : {MODEL_FILENAME_CANDIDATES}")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

pipeline = load_model(model_path)
st.success(f"✅ Modèle chargé: {os.path.basename(model_path)}")

# -------- Charger CSV pour construire le formulaire --------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    return df

df = load_data()
if TARGET_COL not in df.columns:
    st.error(f"❌ Colonne cible '{TARGET_COL}' absente du CSV.")
    st.stop()

X_all = df.drop(columns=[TARGET_COL])
num_cols = X_all.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = X_all.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

num_medians = X_all[num_cols].median(numeric_only=True)
cat_modes = X_all[cat_cols].astype(str).mode().iloc[0] if cat_cols else pd.Series(dtype=str)

st.markdown("### ✍️ Renseigne les caractéristiques")
inputs = {}

# Numériques
st.subheader("Variables numériques")
for c in num_cols:
    default_val = float(num_medians.get(c, 0.0))
    # Ajuster des bornes raisonnables si besoin
    inputs[c] = st.number_input(c, value=default_val)

# Catégorielles
if cat_cols:
    st.subheader("Variables catégorielles")
    for c in cat_cols:
        options = sorted(df[c].dropna().astype(str).unique().tolist())
        default = str(cat_modes.get(c, options[0] if options else ""))
        if options:
            inputs[c] = st.selectbox(c, options, index=options.index(default) if default in options else 0)
        else:
            inputs[c] = default

# -------- Prédiction --------
threshold = st.slider("Seuil de décision (sur la probabilité classe 1)", 0.0, 1.0, 0.5, 0.01)
if st.button("🔮 Prédire"):
    # Construire une ligne avec les colonnes originales
    row = {}
    for c in X_all.columns:
        v = inputs.get(c, None)
        row[c] = v
    X_input = pd.DataFrame([row])

    # Prédiction (le pipeline gère imput, OHE, scale)
    try:
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = float(pipeline.predict_proba(X_input)[0, 1])
        pred = int(proba >= threshold) if proba is not None else int(pipeline.predict(X_input)[0])

        st.markdown("---")
        if pred == 1:
            st.success("🟢 Le modèle prédit que **le client ACHÈTERA** la maison.")
        else:
            st.error("🔴 Le modèle prédit que **le client N'ACHÈTERA PAS** la maison.")

        if proba is not None:
            st.info(f"Probabilité (classe 1) : **{proba*100:.2f} %**  |  Seuil = {int(threshold*100)} %")

        with st.expander("Debug (facultatif)"):
            st.write("Colonnes envoyées au pipeline:", list(X_input.columns))
            st.write("Aperçu ligne:", X_input)

    except Exception as e:
        st.exception(e)
