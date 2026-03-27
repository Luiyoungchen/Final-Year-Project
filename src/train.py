from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

CLEAN_DATA_PATH = PROCESSED_DIR / "clean_pgi_data.csv"
DISTRICT_MAP_PATH = PROCESSED_DIR / "district_encoding_map.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "sample_predictions.csv"
MODEL_PATH = MODEL_DIR / "pgi_model.pkl"


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    district_map = pd.read_csv(DISTRICT_MAP_PATH)
    return df, district_map


# -------------------------------
# FEATURE ENHANCEMENT
# -------------------------------
def enhance_features(df):
    df["Overall_avg"] = (df["Overall_lag1"] + df["Overall_lag2"]) / 2
    df["Outcome_avg"] = (df["Outcome_lag1"] + df["Outcome_lag2"]) / 2

    df["Overall_diff"] = abs(df["Overall_lag1"] - df["Overall_lag2"])
    df["Outcome_diff"] = abs(df["Outcome_lag1"] - df["Outcome_lag2"])

    return df


# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_model(df):

    df = enhance_features(df)

    features = [
        "Overall_lag1", "Overall_lag2", "Overall_trend", "Overall_avg", "Overall_diff",
        "Outcome_lag1", "Outcome_lag2", "Outcome_trend", "Outcome_avg", "Outcome_diff",
        "ECT_lag1", "ECT_lag2",
        "IFSE_lag1", "IFSE_lag2",
        "SSCP_lag1", "SSCP_lag2",
        "DL_lag1", "DL_lag2",
        "GP_lag1", "GP_lag2",
        "Year"
        # 🚨 Removed District_Encoded (noise reduction)
    ]

    target = "Grade_Group"

    # -------------------------------
    # TIME SPLIT
    # -------------------------------
    train = df[df["Year"] < 2022]
    test = df[df["Year"] >= 2022]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # -------------------------------
    # FINAL MODEL
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model Accuracy: {acc:.3f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📊 Feature Importance:")
    for name, importance in zip(features, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    return model, X_test, y_test, pd.Series(y_pred, index=y_test.index)


# -------------------------------
# BUILD OUTPUT
# -------------------------------
def build_predictions(X_test, y_test, y_pred, district_map):

    result = X_test.copy()
    result["Actual_Grade"] = y_test
    result["Predicted_Grade"] = y_pred

    readable = result.merge(
        district_map,
        left_index=True,
        right_on="District_Encoded",
        how="left"
    )

    cols = [
        "District",
        "Year",
        "Actual_Grade",
        "Predicted_Grade"
    ]

    return readable[cols].reset_index(drop=True)


# -------------------------------
# SAVE MODEL
# -------------------------------
def save_model(model):
    import joblib

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved at: {MODEL_PATH}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    df, district_map = load_data()

    model, X_test, y_test, y_pred = train_model(df)

    predictions = build_predictions(X_test, y_test, y_pred, district_map)

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    save_model(model)

    print(f"\n📁 Predictions saved at: {PREDICTIONS_PATH}")
    print("\n🔍 Sample Predictions:")
    print(predictions.head(10))


if __name__ == "__main__":
    main()