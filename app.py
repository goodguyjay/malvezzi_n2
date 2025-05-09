import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eda import explore_data
from models.catboost_model import train_catboost
from models.forest_model import train_random_forest
from models.knn_model import train_knn
from models.lightgbm_model import train_lightgbm
from models.logistic_model import train_logistic
from models.xgb_model import train_xgb
from preprocessing import load_and_split, FeatureEngineer

st.set_page_config(page_title="previsão de Inadimplência", layout="wide")
st.title("previsão de Inadimplência de Clientes")

csv_path = "credit_data.csv"
df = explore_data(csv_path)

X_train, X_test, y_train, y_test = load_and_split(df)

pipeline = Pipeline([
    ("feature_engineer", FeatureEngineer()),
    ("scaler", StandardScaler()),
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

st.sidebar.header("opções")
model_option = st.sidebar.selectbox(
    "escolha o modelo de classificação:",
    ("regressão logística", "random forest", "XGBoost", "catboost", "knn", "lightgbm"),
)


def plot_results(y_test, y_pred, y_prob):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("matriz de confusão")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        st.pyplot(fig)

    with col2:
        st.subheader("curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('taxa de falso positivo')
        ax.set_ylabel('taxa de verdadeiro positivo')
        ax.legend()
        st.pyplot(fig)

    st.subheader("relatório de classificação")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


if model_option == "regressão logística":
    st.subheader("resultados da regressão logística")
    model, y_pred, y_prob = train_logistic(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados da regressão logística")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")

elif model_option == "random forest":
    st.subheader("resultados do random forest")
    model, y_pred, y_prob = train_random_forest(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados do random forest")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")

elif model_option == "XGBoost":
    st.subheader("resultados do XGBoost")
    model, y_pred, y_prob = train_xgb(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados do XGBoost")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")

elif model_option == "catboost":
    st.subheader("resultados do catboost")
    model, y_pred, y_prob = train_catboost(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados do catboost")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")

elif model_option == "knn":
    st.subheader("resultados do knn")
    neighbors = st.sidebar.slider("n_neighbors", min_value=1, max_value=20, value=5)
    model, y_pred, y_prob = train_knn(X_train, X_test, y_train, y_test, neighbors)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados do knn")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")

elif model_option == "lightgbm":
    st.subheader("resultados do lightgbm")
    model, y_pred, y_prob = train_lightgbm(X_train, X_test, y_train, y_test)
    plot_results(y_test, y_pred, y_prob)

    st.subheader("resultados do lightgbm")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2f}")
    col2.metric("Precisão", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    col5.metric("AUC-ROC", f"{auc:.2f}")
