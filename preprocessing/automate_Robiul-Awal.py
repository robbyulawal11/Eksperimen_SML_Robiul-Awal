import numpy as np
import pandas as pd

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    print("Missing values before preprocessing:")
    print(df.isnull().sum())

    imputer = SimpleImputer(strategy="median")
    columns_to_process = ["INFLOW"]

    for col in columns_to_process:
        df[col] = imputer.fit_transform(df[[col]])

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    for col in columns_to_process:
        df = remove_outliers(df, col)

    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())

    return df


def prepare_lstm_data(df, look_back=168, target_column="INFLOW", output_steps=168):
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    feature_columns = [
        "INFLOW",
        "hour",
        "day_of_week",
        "month",
    ]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    joblib.dump(scaler, f"scaler_{target_column}.pkl")

    X, y = [], []
    target_col_index = feature_columns.index(target_column)

    for i in range(len(scaled_data) - look_back - output_steps):
        X.append(scaled_data[i : i + look_back])
        y.append(
            scaled_data[i + look_back : i + look_back + output_steps, target_col_index]
        )

    X = np.array(X)
    y = np.array(y)
    y = y.reshape(
        (y.shape[0], y.shape[1], 1)
    )  # Make target shape compatible with output

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


def main(file_path, target_columns=["INFLOW"]):
    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    for target_column in target_columns:
        print(f"\n--- Preprocessing data {target_column} ---")
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
            df, target_column=target_column
        )

        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        y_train_2d = y_train.reshape(y_train.shape[0], -1)
        y_test_2d = y_test.reshape(y_test.shape[0], -1)

        import os
        os.makedirs("preprocessing/inflow_preprocessing", exist_ok=True)
        pd.DataFrame(X_train_2d).to_csv("preprocessing/inflow_preprocessing/X_train.csv", index=False)
        pd.DataFrame(X_test_2d).to_csv("preprocessing/inflow_preprocessing/X_test.csv", index=False)
        pd.DataFrame(y_train_2d).to_csv("preprocessing/inflow_preprocessing/y_train.csv", index=False)
        pd.DataFrame(y_test_2d).to_csv("preprocessing/inflow_preprocessing/y_test.csv", index=False)
        print("The preprocessing result data has been saved to preprocessing/inflow_preprocessing")


if __name__ == "__main__":
    main("dataset/inflow.csv")
