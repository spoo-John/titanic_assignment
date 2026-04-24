import pandas as pd
import numpy as np

def clean_data(input_path, output_path):

    df = pd.read_csv(input_path)

    # Missing value indicators
    df["AgeMissing"] = df["Age"].isnull().astype(int)
    df["CabinMissing"] = df["Cabin"].isnull().astype(int)

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Fix text inconsistencies
    df["Sex"] = df["Sex"].str.lower().str.strip()
    df["Embarked"] = df["Embarked"].str.upper().str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle outliers in Fare
    Q1 = df["Fare"].quantile(0.25)
    Q3 = df["Fare"].quantile(0.75)

    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR

    df["Fare"] = np.where(df["Fare"] > upper, upper, df["Fare"])

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data("../data/train.csv", "../data/train_cleaned.csv")