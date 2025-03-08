import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Title and Description
st.title("Divorca - AI Divorce Predictor")
st.write("Predict the likelihood of divorce based on relationship factors.")

# File Uploader
df = None
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")

if df is not None:
    # Preprocess Data
    X = df.drop("Divorce Probability", axis=1)
    y = (df["Divorce Probability"] > 2.0).astype(int)  # Binary classification: 1 = High Risk, 0 = Low Risk

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Sidebar - User Inputs
    st.sidebar.header("Answer these to predict your risk")
    user_input = []
    for col in X.columns:
        value = st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        user_input.append(value)

    user_data = np.array(user_input).reshape(1, -1)

    # Prediction
    prediction = model.predict(user_data)[0]
    prediction_prob = model.predict_proba(user_data)[0]

    # Output Result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Divorce")
    else:
        st.success("✅ Low Risk of Divorce")

    st.write(f"Confidence: {prediction_prob[prediction]*100:.2f}%")

    # Feature Importance using SHAP (If SHAP is available)
    st.subheader("Why this prediction? (Feature Importance)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.title("SHAP Feature Importance")
    st.pyplot(fig)
    
    # Ethics Section
    st.subheader("Ethical Considerations")
    st.write(
        "This tool is intended for academic purposes only. Predicting relationship outcomes can be sensitive, "
        "and such predictions should never replace professional counseling. Bias in data can lead to unfair or "
        "inaccurate predictions."
    )

    # Model Performance (optional for defense)
    if st.checkbox("Show Model Performance"):
        y_pred = model.predict(X_test)
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))