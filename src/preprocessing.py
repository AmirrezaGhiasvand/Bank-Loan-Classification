import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    # Fix negative experience values
    df["Experience"] = df["Experience"].abs()

    # Remove unnecessary ID column if present
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    return df


def split_features_target(df):

    X = df.drop("Personal Loan", axis=1)

    y = df["Personal Loan"]

    return X, y