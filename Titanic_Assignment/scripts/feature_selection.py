import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features(input_path):
    df = pd.read_csv(input_path)

    drop_cols = []
    for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=["Survived"] + drop_cols)
    y = df["Survived"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Top Features:")
    print(importance_df.head(15))

    selected = importance_df[importance_df["Importance"] > 0.01]["Feature"].tolist()
    print("\nSelected Features:")
    print(selected)

if __name__ == "__main__":
    select_features("data/train_featured.csv")