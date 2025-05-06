from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)


def train_models(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    treina e avalia dois modelos de classificação: regressão logística e Random Forest.
    :param x_train: Dados de treinamento
    :param x_test: dados de teste
    :param y_train: feedback de treinamento
    :param y_test: feedback de teste
    :return: None
    """

    print("regressão logística")
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x_train, y_train)
    y_pred_logistic = logistic_model.predict(x_test)

    print("avaliação - regressão logística")
    evaluate_model(y_test, y_pred_logistic)

    print("Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)

    print("avaliação - Random Forest")
    evaluate_model(y_test, y_pred_rf)

    print("curvas ROC (logística vs Random Forest)")
    plot_roc_curves(y_test, logistic_model, rf_model, x_test)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    mostra as métricas de avaliação do modelo e a matriz de confusão
    :param y_true: valores reais
    :param y_pred: valores previstos
    :return: None
    """

    acc = accuracy_score(y_true, y_pred)
    print(f"acurácia: {acc:.4f}\n")

    print("relatório de classificação:")
    print(classification_report(y_true, y_pred))

    print("matriz de confusão:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("previsão")
    plt.ylabel("real")
    plt.title("matriz de confusão")
    plt.show()
    plt.close()


def plot_roc_curves(y_true: np.ndarray, model1: Any, model2: Any, x_test: np.ndarray) -> None:
    """
    Plota as curvas ROC comparando dois modelos
    :param y_true: valores real
    :param model1: primeiro modelo treinado
    :param model2: segundo modelo treinado
    :param x_test: dados de teste
    :return: None
    """

    y_score1 = model1.predict_proba(x_test)[:, 1]
    y_score2 = model2.predict_proba(x_test)[:, 1]

    fpr1, tpr1, _ = roc_curve(y_true, y_score1)
    fpr2, tpr2, _ = roc_curve(y_true, y_score2)

    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    plt.plot(fpr1, tpr1, label=f"Logística (AUC = {auc1:.4f})")
    plt.plot(fpr2, tpr2, label=f"Random Forest (AUC = {auc2:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel('falso positivo')
    plt.ylabel('verdadeiro positivo')
    plt.title('curvas ROC')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
