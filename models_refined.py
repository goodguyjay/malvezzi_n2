from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, accuracy_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV

RANDOM_STATE: int = 42


def refine_models(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    refina o LogisticRegression e o RandomForestClassifier com GridSearchCV e compara os resultados
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """

    # regressão logística ->
    print("regressão logística (base)")
    base_log = LogisticRegression(max_iter=1000)
    base_log.fit(X_train, y_train)
    y_pred_log_base = base_log.predict(X_test)
    _evaluate("logística (base)", y_test, y_pred_log_base)

    # otimização - regressão logística c/ grid search
    param_grid_log: Dict[str, Any] = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"],
    }
    grid_log = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid_log,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_log.fit(X_train, y_train)

    print("melhores parâmetros - regressão logística:", grid_log.best_params_)

    y_pred_log_grid = grid_log.predict(X_test)
    _evaluate("logística (grid search)", y_test, y_pred_log_grid)

    # random forest ->
    print("random forest (base)")
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    base_rf.fit(X_train, y_train)
    y_pred_rf_base = base_rf.predict(X_test)
    _evaluate("random forest (padrão)", y_test, y_pred_rf_base)

    # "otimização - random forest c/ grid search"
    param_grid_rf: Dict[str, Any] = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid_rf,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_rf.fit(X_train, y_train)

    print("melhores parâmetros - random forest:", grid_rf.best_params_)

    y_pred_rf_grid = grid_rf.predict(X_test)
    _evaluate("random forest (grid search)", y_test, y_pred_rf_grid)

    # comparação de modelos
    print("curvas ROC (logística vs Random Forest)")
    _plot_roc_curves(y_test, grid_log.best_estimator_, grid_rf.best_estimator_, X_test)


# helpers
def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    avalia um modelo com métricas clássicas e matriz de confusão
    :param name: nome do modelo
    :param y_true: valores reais
    :param y_pred: valores previstos
    :return: None
    """

    print(f"\n{name}")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    # print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"matriz de confusão - {name}")
    plt.xlabel("previsão")
    plt.ylabel("real")
    plt.show()
    plt.close()


def _plot_roc_curves(y_true: np.ndarray,
                     model1: Any,
                     model2: Any,
                     X_test: np.ndarray, ) -> None:
    """
    plota as curvas ROC para os modelos de regressão logística e Random Forest
    :param y_true: valores reais
    :param model1: modelo de regressão logística
    :param model2: modelo Random Forest
    :param X_test: dados de teste
    :return: None
    """

    y_score1 = model1.predict_proba(X_test)[:, 1]
    y_score2 = model2.predict_proba(X_test)[:, 1]

    fpr1, tpr1, _ = roc_curve(y_true, y_score1)
    fpr2, tpr2, _ = roc_curve(y_true, y_score2)

    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    plt.plot(fpr1, tpr1, label=f"logístico (AUC {auc1:.3f})")
    plt.plot(fpr2, tpr2, label=f"random Forest (AUC {auc2:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="chute")
    plt.xlabel("falso positivos")
    plt.ylabel("positivos reais")
    plt.title("ROC – modelos refinados")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
