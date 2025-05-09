from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # drop ID se existir
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        # renomeia target antes de chamar transform (se preferir)
        # Médias e desvios
        for prefix in ('BILL_AMT', 'PAY_AMT'):
            cols = [f'{prefix}{i}' for i in range(1, 7)]
            df[f'mean_{prefix.lower()}'] = df[cols].mean(axis=1)
            df[f'std_{prefix.lower()}'] = df[cols].std(axis=1)

        all_delay_cols = [f"PAY_{i}" for i in range(0, 7)]  # PAY_0 ... PAY_6
        cols = [c for c in all_delay_cols if c in df.columns]

        df["mean_delay"] = df[cols].mean(axis=1)
        df["max_delay"] = df[cols].max(axis=1)
        df["has_late"] = df[cols].gt(0).any(axis=1).astype(int)

        # razões
        df['bill_limit_ratio'] = df['mean_bill_amt'] / df['LIMIT_BAL']
        df['pay_bill_ratio'] = df['mean_pay_amt'] / (df['mean_bill_amt'] + 1e-6)

        # flags
        df['has_late'] = df[cols].gt(0).any(axis=1).astype(int)
        df['recently_paid'] = df['PAY_AMT1'].gt(0).astype(int)

        return df


def load_and_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.rename(columns={'default.payment.next.month': 'default'}) \
        if 'default.payment.next.month' in df.columns else df.rename(columns={'default': 'default'})
    X = df.drop(columns=['default'])
    y = df['default'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
