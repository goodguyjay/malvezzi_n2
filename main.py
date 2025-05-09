from eda import explore_data
from forest import train_random_forest
from logistic import train_logistic
from preprocessing import prepare_data
from xgb import train_xgb
from app import *

if __name__ == "__main__":
    CSV_PATH = "credit_data.csv"

    df = explore_data(CSV_PATH)
    x_train, x_test, y_train, y_test = prepare_data(df)

    train_logistic(x_train, x_test, y_train, y_test)  # refinado
    train_random_forest(x_train, x_test, y_train, y_test)
    train_xgb(x_train, x_test, y_train, y_test)  # boosted
