from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from core import FEATURES, FORMATS, STEPS, UNITS, molecular_so2, validate

MODEL_PATH = Path("model/wine_rf.joblib")

st.set_page_config(page_title="Wine Quality Screening", page_icon="🍷", layout="wide")

if not MODEL_PATH.exists():
    with st.spinner("Training the model for the first time (about a minute)..."):
        from train_model import main as train_model
        train_model()


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


artifacts = load_artifacts()
model, meta = artifacts["model"], artifacts["meta"]
ranges = meta["ranges"]

PRESETS = {"Typical red wine": "red", "Typical white wine": "white"}


def apply_preset():
    choice = st.session_state["preset"]
    if choice in PRESETS:
        for feature in FEATURES:
            st.session_state[f"in_{feature}"] = float(meta["type_medians"][PRESETS[choice]][feature])


for feature in FEATURES:
    st.session_state.setdefault(f"in_{feature}", float(ranges[feature]["median"]))

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Laboratory profile")
st.sidebar.selectbox("Start from", ["Custom", *PRESETS], key="preset", on_change=apply_preset)

values = {}
for feature in FEATURES:
    unit = f" ({UNITS[feature]})" if UNITS[feature] else ""
    values[feature] = st.sidebar.slider(
        f"{feature.capitalize()}{unit}",
        float(ranges[feature]["min"]),
        float(ranges[feature]["max"]),
        step=STEPS[feature],
        format=FORMATS[feature],
        key=f"in_{feature}",
    )

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Screening threshold", 0.05, 0.95, round(meta["best_threshold"], 2), 0.01,
    help="Lower = more batches flagged and more high-quality wines found, but more false alarms.",
)

operating = meta["operating_points"]
op_row = operating.iloc[(operating["threshold"] - threshold).abs().argmin()]

# ---------------------------------------------------------------- main
st.title("🍷 Wine Quality Screening Tool")
st.caption(
    "Vinho Verde red and white wines · Random Forest trained on laboratory measurements · "
    "educational portfolio project"
)

tab_screen, tab_model = st.tabs(["Screen a wine", "About the model"])

with tab_screen:
    problems = validate(values)
    if problems:
        for message in problems:
            st.error(message)
    else:
        X_in = pd.DataFrame([values])[FEATURES]
        score = float(model.predict_proba(X_in)[0, 1])
        flagged = score >= threshold

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Model score", f"{score:.2f}",
            help="Output of the model between 0 and 1. It is a ranking score, not a calibrated probability "
                 "(the model uses balanced class weights).",
        )
        c2.metric("Screening decision", "Send to sensory panel" if flagged else "Routine")
        c3.metric(
            "Molecular SO₂", f"{molecular_so2(values['free sulfur dioxide'], values['pH']):.2f} mg/L",
            help="Active fraction of free SO₂ at this pH (pKa₁ = 1.81).",
        )
        st.progress(min(max(score, 0.0), 1.0))

        if flagged:
            st.success(
                f"Score {score:.2f} ≥ threshold {threshold:.2f}: this profile resembles wines that "
                "tasters rated 7 or higher. It would be prioritised for sensory evaluation."
            )
        else:
            st.info(
                f"Score {score:.2f} < threshold {threshold:.2f}: this profile does not resemble the "
                "high-quality wines closely enough to prioritise it."
            )

        st.subheader("How does this profile compare with the training data?")
        comparison = pd.DataFrame({
            "Your wine": pd.Series(values),
            "Median, high quality (≥ 7)": pd.Series(meta["medians_high"]),
            "Median, other wines": pd.Series(meta["medians_rest"]),
        }).loc[FEATURES].round(4)
        st.dataframe(comparison)

    st.caption(
        "The model captures statistical **associations** in a historical dataset. Changing a slider shows how "
        "the score responds, not how a real wine would change. This tool does not replace sensory evaluation."
    )

with tab_model:
    st.subheader("Performance on the held-out test set")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC", f"{meta['test']['pr_auc']:.2f}", help="Average precision (threshold-independent).")
    m2.metric("No-skill PR-AUC", f"{meta['test']['baseline_pr_auc']:.2f}", help="Share of high-quality wines.")
    m3.metric("ROC-AUC", f"{meta['test']['roc_auc']:.2f}")
    m4.metric("Test wines", f"{meta['n_test']:,}")

    st.markdown(
        f"**At threshold {threshold:.2f}:** {op_row['flagged_pct']:.0f}% of batches are flagged, "
        f"{op_row['precision']:.0%} of the flagged batches are high quality (precision) and "
        f"{op_row['recall']:.0%} of all high-quality wines are found (recall)."
    )

    st.subheader("Variable importance (impurity-based)")
    st.bar_chart(pd.Series(meta["importances"]).sort_values(), horizontal=True)

    st.markdown(
        """
**Data:** Cortez et al. (2009), *Modeling wine preferences by data mining from physicochemical properties*,
Decision Support Systems 47(4). Vinho Verde wines; quality is the median of at least three tasters.

**Limitations:** observational data from one region and period; high quality is defined as score ≥ 7;
correlated predictors (alcohol, sugar, density) share importance; the screening threshold rests on an
illustrative cost assumption.
        """
    )
    st.caption(f"Trained with scikit-learn {meta['sklearn_version']} on {meta['n_train']:,} wines.")
