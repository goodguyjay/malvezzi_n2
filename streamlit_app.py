import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocessing import prepare_data
from models import train_models


# Load and preprocess
@st.cache_data
def load_and_train():
    df = pd.read_csv("credit_data.csv")

    # Remove ID if exists
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Rename target if needed
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})

    X = df.drop(columns=['default'])
    y = df['default']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)

    return X.columns.tolist(), scaler, rf_model


# App
st.title("📉 Credit Default Prediction App")

columns, scaler, model = load_and_train()

st.sidebar.header("🧪 Enter Client Data")

# Coletar dados manualmente
client_data = []
for col in columns:
    # Define valores padrão só pra não bugar visualmente
    val = st.sidebar.number_input(f"{col}", value=0.0)
    client_data.append(val)

client_array = np.array(client_data).reshape(1, -1)
client_scaled = scaler.transform(client_array)

# Previsão
prediction = model.predict(client_scaled)[0]
probability = model.predict_proba(client_scaled)[0][1]

st.subheader("🔍 Prediction")
st.write("**Result:**", "❌ Will Default" if prediction == 1 else "✅ Will Not Default")
st.write(f"**Probability of Default:** {probability:.2%}")
