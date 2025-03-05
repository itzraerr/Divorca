import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Title and Description
st.title("Divorca - AI Divorce Predictor")
st.write("Predict the likelihood of divorce based on relationship factors.")

# Load dataset (South Africa dataset you provided)
@st.cache_data
def load_data():
    df = pd.read_csv('data/divorce_data.csv')
    return df

df = load_data()

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

# Feature Importance Explanation (Instead of SHAP)
st.subheader("Why this prediction? (Feature Importance)")
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Feature')
ax.set_title('Feature Importance in Prediction')
st.pyplot(fig)

# Ethics Section
st.subheader("Ethical Considerations")
st.write(
    "This tool is intended for academic purposes only. Predicting relationship outcomes can be sensitive, and such predictions should never replace professional counseling. Bias in data can lead to unfair or inaccurate predictions."
)

# Model Performance (optional for defense)
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))