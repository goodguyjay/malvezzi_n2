from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_logistic(X_train, X_test, y_train, y_test) -> Tuple:
    print(f"{Fore.CYAN}\ntreinando regressão logística...{Style.RESET_ALL}")

    model = LogisticRegression(C=10, penalty='l2', solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{Fore.GREEN}acurácia: {accuracy_score(y_test, y_pred):.4f}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}relatório de classificação:{Style.RESET_ALL}\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("matriz de confusão - regressão logística")
    plt.xlabel("predito")
    plt.ylabel("verdadeiro")
    plt.show()

    return model, y_pred, y_prob
