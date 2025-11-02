import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import streamlit as st
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from PIL import Image
import io

# load model + data
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# recall value on original data with model
recall_failure_class = 0.988479262672811
# Load feature names
feature_names = joblib.load("feature_names.pkl")
fT_df = pd.read_csv("feature_importance.csv")
cm = np.load("confusion_matrix.npy")
y_test = np.load("y_test.npy")
y_pred = np.load("y_pred.npy")

# display project summary
st.title("Manufacturing Failure Prediction")
st.markdown("""
This dashboard shows how the trained XGBoost model detects failures in manufacturing sensor data.
It uses SMOTE oversampling to handle class imbalance and grid search tuning for optimization.
Below you can explore the model’s feature importance, confusion matrix, and there is a place to upload future
updated datasets to find feature importance again.
""")

# feature importance from original data
st.header("Top 20 Most Important Features")
top_n = 20
top_features = fT_df.head(top_n)

# show cm and feature importance
st.write("""
These features had the highest impact on whether a product passed or failed inspection.
This helps identify which sensors are most influential for predicting equipment faults.
""")
sb.heatmap(cm, annot=True, cmap="Blues")

# model Visualization Section
st.header("Model Insights from Original + testing data.")

# --- NEW IMAGE LOADING APPROACH ---
def load_image_bytes(filename: str):
    img_path = Path(__file__).parent / filename
    with open(img_path, "rb") as f:
        return f.read()

feature_importance_bytes = load_image_bytes("feature_importance.png")
confusion_matrix_bytes = load_image_bytes("confusion_matrix.png")

st.subheader("Top Model Features")
st.image(Image.open(io.BytesIO(feature_importance_bytes)), caption="Feature Importance (XGBoost Model)", use_container_width=True)

st.subheader("Model Performance")
st.image(Image.open(io.BytesIO(confusion_matrix_bytes)), caption="Confusion Matrix - Model Accuracy & Recall", use_container_width=True)

# progress bar visualizing recall accuracy
st.markdown(f"**Model Recall (Failure Class): {recall_failure_class:.2%}**")
st.progress(recall_failure_class)
st.caption(
    "High recall (close to 100%) means the model rarely misses actual failures — "
    "critical for minimizing production defects."
)
st.write("""
These features had the highest impact on whether a product passed or failed inspection.
This helps identify which sensors are most influential for predicting equipment faults, improving
defect reporting efficiency in the manufacturing process.
""")

st.subheader("Upload New P/f Manufacturing data")
st.write(
    "Upload a CSV file containing the same features the model was trained on. "
    "The file should **not** include the target column (e.g., 'Pass/Fail')."
)

# --- NEW CSV UPLOAD APPROACH ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # read CSV from bytes
        content = uploaded_file.read()
        user_df = pd.read_csv(io.BytesIO(content))
        st.success(f"Loaded {uploaded_file.name} ({user_df.shape[0]} rows, {user_df.shape[1]} cols)")

        if user_df.isnull().any().any():
            st.warning("Missing values filled with 0")
            user_df = user_df.fillna(0)

        target_col = None
        for col in user_df.columns:
            if col.lower().strip() in ["pass/fail", "pass_fail", "target", "label"]:
                target_col = col
                break

        if target_col:
            st.info(f"Detected target col: {target_col}")
            y_true = user_df[target_col].replace({-1: 0, 1: 1}).astype(int)
            X_new = user_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        else:
            y_true = None
            X_new = user_df.select_dtypes(include=[np.number])

        # align, scale
        for col in feature_names:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[feature_names]
        X_scaled = scaler.transform(X_new)

        with st.spinner("Running model predictions..."):
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]
            st.success("Predictions done")

        result_df = user_df.copy()
        result_df["Predicted Pass/Fail"] = preds
        result_df["Probability (Fail)"] = probs
        st.write("### Prediction Results")
        st.dataframe(result_df.head(10))

        if y_true is not None:
            cm_new = confusion_matrix(y_true, preds)
            report = classification_report(y_true, preds, output_dict=True)
            st.subheader("Evaluation on Uploaded Data")
            st.write(f"**Recall (Failure): {report['1']['recall']:.2%}**")
            st.write(f"**Accuracy:** {report['accuracy']:.2%}")
            fig, ax = plt.subplots()
            sb.heatmap(cm_new, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix - Uploaded Data")
            st.pyplot(fig)

        st.subheader("Top 20 Model Features")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_importance(model, max_num_features=20, importance_type='weight', ax=ax, grid=False, show_values=True)
        ax.set_title("Top 20 Feature Importances", fontsize=14)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("[GitHub Repo](https://github.com/alundy98/SECOM-Manufacturing-Fail-Case-Predictive-Model)")
