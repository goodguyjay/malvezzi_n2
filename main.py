from eda import explore_data
from models import train_models
from models_boosted import boosted_grid
from models_refined import refine_models
from preprocessing import prepare_data

if __name__ == "__main__":
    CSV_PATH = "credit_data.csv"

    df = explore_data(CSV_PATH)
    x_train, x_test, y_train, y_test = prepare_data(df)

    train_models(x_train, x_test, y_train, y_test)  # base
    refine_models(x_train, x_test, y_train, y_test)  # refinado
    boosted_grid(x_train, x_test, y_train, y_test)  # boosted
