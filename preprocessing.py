from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    limpa e prepara os dados para o treinamento do modelo.
    :param df: Dataframe com os dados brutos
    :return: x_train, X_test, y_train, y_test (todos como arrays)
    """

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})

    X = df.drop(columns=['default'])
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                                        stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()
