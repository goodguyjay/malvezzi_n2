from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, X_test, y_train, y_test, neighbors: int = 5) -> Tuple:
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"knn (k={neighbors}) - Accuracy:", accuracy_score(y_test, y_pred))
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Oranges', fmt='d')
    plt.title(f"matriz de confusão - KNN (k={neighbors})")
    plt.show()

    return model, y_pred, y_prob
