import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Derived features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title extraction
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                   "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Deck extraction
    df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "Unknown")

    # Age groups
    def age_group(age):
        if age < 13:
            return "Child"
        elif age < 20:
            return "Teen"
        elif age < 60:
            return "Adult"
        else:
            return "Senior"

    df["AgeGroup"] = df["Age"].apply(age_group)

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Transformations
    df["FareLog"] = np.log1p(df["Fare"])
    df["AgeLog"] = np.log1p(df["Age"])

    # Optional interactions
    df["Pclass_Fare"] = df["Pclass"] * df["Fare"]

    # One-hot encoding
    categorical_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to {output_path}")

if __name__ == "__main__":
    engineer_features("data/train_cleaned.csv", "data/train_featured.csv")