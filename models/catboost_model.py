from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_catboost(X_train, X_test, y_train, y_test) -> Tuple:
    model = CatBoostClassifier(
        depth=8,  # 6–10
        learning_rate=0.05,  # 0.03–0.1
        l2_leaf_reg=3,  # 1–10
        iterations=1500,
        random_seed=42,
        verbose=False,
        loss_function="Logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("CatBoost - acuracia:", accuracy_score(y_test, y_pred))
    print("relatório de classificação:\n", classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Greens', fmt='d')
    plt.title("matriz de confusão - CatBoost")
    plt.show()

    return model, y_pred, y_prob
