from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = PROCESSED_DIR / "clean_pgi_data.csv"
MODEL_PATH = MODEL_DIR / "pgi_model.pkl"
OUTPUT_PATH = PROCESSED_DIR / "predictions_2024.csv"


# -------------------------------
# LOAD MODEL + DATA
# -------------------------------
def load():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    return df, model


# -------------------------------
# CREATE LATEST FEATURES (2023)
# -------------------------------
def create_features(df):
    
    # Get last 2 years per district
    df = df.sort_values(["District_Encoded", "Year"])

    latest_rows = []

    for district in df["District_Encoded"].unique():
        d = df[df["District_Encoded"] == district]

        if len(d) < 2:
            continue

        last = d.iloc[-1]     # 2023
        prev = d.iloc[-2]     # 2022

        row = {
            "District_Encoded": district,
            "Year": 2024,

            # lag features
            "Overall_lag1": last["Overall_lag1"],
            "Overall_lag2": prev["Overall_lag1"],

            "Outcome_lag1": last["Outcome_lag1"],
            "Outcome_lag2": prev["Outcome_lag1"],

            "ECT_lag1": last["ECT_lag1"],
            "ECT_lag2": prev["ECT_lag1"],

            "IFSE_lag1": last["IFSE_lag1"],
            "IFSE_lag2": prev["IFSE_lag1"],

            "SSCP_lag1": last["SSCP_lag1"],
            "SSCP_lag2": prev["SSCP_lag1"],

            "DL_lag1": last["DL_lag1"],
            "DL_lag2": prev["DL_lag1"],

            "GP_lag1": last["GP_lag1"],
            "GP_lag2": prev["GP_lag1"],
        }

        # derived features
        row["Overall_trend"] = row["Overall_lag1"] - row["Overall_lag2"]
        row["Outcome_trend"] = row["Outcome_lag1"] - row["Outcome_lag2"]

        row["Overall_avg"] = (row["Overall_lag1"] + row["Overall_lag2"]) / 2
        row["Outcome_avg"] = (row["Outcome_lag1"] + row["Outcome_lag2"]) / 2

        row["Overall_diff"] = abs(row["Overall_lag1"] - row["Overall_lag2"])
        row["Outcome_diff"] = abs(row["Outcome_lag1"] - row["Outcome_lag2"])

        latest_rows.append(row)

    return pd.DataFrame(latest_rows)


# -------------------------------
# PREDICT
# -------------------------------
def predict(df_features, model):

    features = [
        "Overall_lag1", "Overall_lag2", "Overall_trend", "Overall_avg", "Overall_diff",
        "Outcome_lag1", "Outcome_lag2", "Outcome_trend", "Outcome_avg", "Outcome_diff",
        "ECT_lag1", "ECT_lag2",
        "IFSE_lag1", "IFSE_lag2",
        "SSCP_lag1", "SSCP_lag2",
        "DL_lag1", "DL_lag2",
        "GP_lag1", "GP_lag2",
        "Year"
    ]

    X = df_features[features]

    df_features["Predicted_Group"] = model.predict(X)

    # convert to readable labels
    label_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    df_features["Predicted_Label"] = df_features["Predicted_Group"].map(label_map)

    return df_features


# -------------------------------
# MAIN
# -------------------------------
def main():
    df, model = load()

    df_features = create_features(df)

    predictions = predict(df_features, model)

    # 🔥 ADD THIS PART HERE
    district_map = pd.read_csv(PROCESSED_DIR / "district_encoding_map.csv")
    predictions = predictions.merge(district_map, on="District_Encoded")

    # Save
    predictions.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ 2024 Predictions Generated!")
    print(predictions[["District", "Predicted_Label"]].head(10))


if __name__ == "__main__":
    main()