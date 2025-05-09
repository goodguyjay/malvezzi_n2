from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from colorama import Fore, Style
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42


def train_xgb(X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
    print(f"{Fore.CYAN}\ntreinando XGBoost com grid search...{Style.RESET_ALL}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cpu",
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=RANDOM_STATE,
    )

    param_grid: Dict[str, Any] = {
        "n_estimators": [400, 800],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5],
        "gamma": [0, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "learning_rate": [0.03, 0.05],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"{Fore.MAGENTA}melhores parâmetros XGBoost:{Style.RESET_ALL} {grid.best_params_}")

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    print(f"{Fore.GREEN}acurácia: {accuracy_score(y_test, y_pred):.4f}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}AUC-ROC : {roc_auc_score(y_test, y_prob):.4f}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}relatório de classificação:{Style.RESET_ALL}\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title("matriz de confusão - XGBoost")
    plt.xlabel("predito")
    plt.ylabel("verdadeiro")
    plt.show()

    return model, y_pred, y_prob
