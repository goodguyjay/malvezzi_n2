from typing import Dict, Any
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42


def boosted_grid(X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Treina XGBoost com busca de hiperparâmetros focada em AUC."""

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # balanceamento
        random_state=RANDOM_STATE,
        use_label_encoder=False
    )

    param_grid: Dict[str, Any] = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
    }

    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    print("melhores parâmetros xgb", grid.best_params_)
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    print(f"acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC : {roc_auc_score(y_test, y_prob):.4f}")
