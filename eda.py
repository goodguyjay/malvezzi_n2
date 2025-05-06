import pandas as pd


def explore_data(csv_path: str) -> pd.DataFrame:
    """
    lê um arquivo CSV e realiza uma exploração inicial dos dados.
    :param csv_path: Path para o arquivo csv
    :return: dataframe carregado
    """

    df = pd.read_csv(csv_path)

    print("primeiras linhas:")
    print(df.head(), "\n")

    print("informações gerais:")
    df.info()

    print("valores nulos por coluna:")
    print(df.isnull().sum(), "\n")

    print("estatísticas descritivas:")
    print(df.describe())

    return df
