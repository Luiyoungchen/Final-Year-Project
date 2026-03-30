from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

CLEAN_DATA_PATH = PROCESSED_DIR / "clean_pgi_data.csv"
DISTRICT_MAP_PATH = PROCESSED_DIR / "district_encoding_map.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "sample_predictions.csv"
MODEL_PATH = MODEL_DIR / "pgi_model.pkl"

GRADE_LABELS = {
    1: "Akanshi",
    2: "Prachesta-1",
    3: "Prachesta-2",
    4: "Prachesta-3",
    5: "Uttam",
    6: "Ati-Uttam",
}


def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    district_map = pd.read_csv(DISTRICT_MAP_PATH)
    return df, district_map


def enhance_features(df):
    df["Overall_avg"] = (df["Overall_lag1"] + df["Overall_lag2"]) / 2
    df["Outcome_avg"] = (df["Outcome_lag1"] + df["Outcome_lag2"]) / 2
    df["Overall_diff"] = abs(df["Overall_lag1"] - df["Overall_lag2"])
    df["Outcome_diff"] = abs(df["Outcome_lag1"] - df["Outcome_lag2"])
    return df


def rebalance_training_data(train, target):
    class_counts = train[target].astype(int).value_counts().sort_index()
    target_count = int(class_counts.median())
    balanced_parts = []

    print("\nOriginal train class counts:")
    print(class_counts.to_string())
    print(f"\nModerate rebalance target: {target_count}")

    for class_value in class_counts.index:
        class_rows = train[train[target].astype(int) == class_value]

        if len(class_rows) < target_count:
            class_rows = resample(
                class_rows,
                replace=True,
                n_samples=target_count,
                random_state=42,
            )

        balanced_parts.append(class_rows)

    balanced_train = pd.concat(balanced_parts, ignore_index=True)
    balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced train class counts:")
    print(balanced_train[target].astype(int).value_counts().sort_index().to_string())

    return balanced_train


def train_model(df):
    df = enhance_features(df)

    features = [
        "Overall_lag1",
        "Overall_lag2",
        "Overall_trend",
        "Overall_avg",
        "Overall_diff",
        "Outcome_lag1",
        "Outcome_lag2",
        "Outcome_trend",
        "Outcome_avg",
        "Outcome_diff",
        "ECT_lag1",
        "ECT_lag2",
        "IFSE_lag1",
        "IFSE_lag2",
        "SSCP_lag1",
        "SSCP_lag2",
        "DL_lag1",
        "DL_lag2",
        "GP_lag1",
        "GP_lag2",
        "Year",
    ]
    target = "Next_Grade"

    train = df[df["Year"] < 2022]
    test = df[df["Year"] >= 2022]
    train = rebalance_training_data(train, target)

    X_train = train[features]
    y_train = train[target].astype(int)
    X_test = test[features]
    y_test = test[target].astype(int)

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nFeature Importance:")
    for name, importance in zip(features, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    return model, y_test, pd.Series(y_pred, index=y_test.index)


def build_predictions(test_rows, y_test, y_pred, district_map):
    result = test_rows.copy()
    result["Actual_Grade_Num"] = y_test.astype(int)
    result["Predicted_Grade_Num"] = y_pred.astype(int)
    result["Actual_Grade_Label"] = result["Actual_Grade_Num"].map(GRADE_LABELS)
    result["Predicted_Grade_Label"] = result["Predicted_Grade_Num"].map(GRADE_LABELS)
    result = result.join(district_map.set_index("District_Encoded"), on="District_Encoded")

    cols = [
        "District",
        "Year",
        "Actual_Grade_Num",
        "Actual_Grade_Label",
        "Predicted_Grade_Num",
        "Predicted_Grade_Label",
    ]
    return result[cols].reset_index(drop=True)


def save_model(model):
    import joblib

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved at: {MODEL_PATH}")


def main():
    df, district_map = load_data()
    model, y_test, y_pred = train_model(df)

    test_rows = df.loc[y_test.index, ["District_Encoded", "Year"]].copy()
    predictions = build_predictions(test_rows, y_test, y_pred, district_map)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    save_model(model)

    print(f"\nPredictions saved at: {PREDICTIONS_PATH}")
    print("\nSample Predictions:")
    print(predictions.head(10))


if __name__ == "__main__":
    main()
