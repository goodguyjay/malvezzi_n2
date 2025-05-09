from typing import Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_lightgbm(X_train, X_test, y_train, y_test) -> Tuple:
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,  # 2 × max_depth ≈ bom ponto de partida
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("relatório de classificação:\n", classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Purples', fmt='d')
    plt.title("matriz de confusão - lightGBM")
    plt.show()

    return model, y_pred, y_prob
