from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_random_forest(X_train, X_test, y_train, y_test) -> Tuple:
    print(f"{Fore.CYAN}\ntreinando random forest...{Style.RESET_ALL}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"{Fore.GREEN}acurácia: {accuracy_score(y_test, y_pred):.4f}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}relatório de classificação:{Style.RESET_ALL}\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("matriz de confusão - random forest")
    plt.xlabel("predito")
    plt.ylabel("verdadeiro")
    plt.show()

    return model, y_pred, y_prob
