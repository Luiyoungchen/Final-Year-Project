import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

GRADE_MAP = {
    "akanshi": 1,
    "akanshi1": 1,
    "akanshi2": 1,
    "prachesta1": 2,
    "prachesta2": 3,
    "prachesta3": 4,
    "utkarsh": 5,
    "uttam": 5,
    "uttam2": 5,
    "uttam3": 5,
    "atiuttam": 6,
}


def read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Cannot read file: {file_path}")


def normalize_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).lower())


def normalize_grade_label(label: str) -> str:
    return normalize_name(str(label))


def find_column(column_map: dict, required_tokens: list[str]) -> str:
    for original_name, normalized_name in column_map.items():
        if all(token in normalized_name for token in required_tokens):
            return original_name
    raise KeyError(f"Column not found: {required_tokens}")


def get_year_from_filename(file_name: str) -> int:
    for year in [2018, 2019, 2020, 2021, 2022, 2023]:
        if str(year) in file_name:
            return year
    raise ValueError(f"Year not found in filename: {file_name}")


def standardize_dataframe(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    column_map = {col: normalize_name(col) for col in df.columns}

    return pd.DataFrame(
        {
            "District": df[find_column(column_map, ["district"])],
            "Grade": df[find_column(column_map, ["grade"])],
            "Overall": df[find_column(column_map, ["overall"])],
            "Outcome": df[find_column(column_map, ["outcome"])],
            "ECT": df[find_column(column_map, ["ect"])],
            "IFSE": df[find_column(column_map, ["ifse"])],
            "SSCP": df[find_column(column_map, ["sscp"])],
            "DL": df[find_column(column_map, ["dl"])],
            "GP": df[find_column(column_map, ["gp"])],
            "Year": get_year_from_filename(file_name),
        }
    )


def main():
    print("Inside main")

    dfs = []

    for file_path in RAW_PATH.glob("*.csv"):
        print("Processing:", file_path)
        raw_df = read_csv_with_fallback(file_path)
        standardized_df = standardize_dataframe(raw_df, file_path.name)
        dfs.append(standardized_df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {RAW_PATH}")

    df = pd.concat(dfs, ignore_index=True)
    print("Combined Data Shape:", df.shape)

    df["Grade_Num"] = df["Grade"].astype(str).map(normalize_grade_label).map(GRADE_MAP).fillna(-1).astype(int)

    le = LabelEncoder()
    df["District_Encoded"] = le.fit_transform(df["District"])

    district_map_df = pd.DataFrame(
        {
            "District": le.classes_,
            "District_Encoded": range(len(le.classes_)),
        }
    )

    numeric_features = ["Overall", "Outcome", "ECT", "IFSE", "SSCP", "DL", "GP"]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_features] = df[numeric_features].fillna(0)

    df = df.sort_values(["District_Encoded", "Year"])
    for col in numeric_features:
        df[f"{col}_lag1"] = df.groupby("District_Encoded")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("District_Encoded")[col].shift(2)

    df["Overall_trend"] = df["Overall_lag1"] - df["Overall_lag2"]
    df["Outcome_trend"] = df["Outcome_lag1"] - df["Outcome_lag2"]
    df["Next_Grade"] = df.groupby("District_Encoded")["Grade_Num"].shift(-1)

    df = df.dropna()
    print("After Feature Engineering:", df.shape)

    final_cols = [
        "Overall_lag1",
        "Overall_lag2",
        "Overall_trend",
        "Outcome_lag1",
        "Outcome_lag2",
        "Outcome_trend",
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
        "District_Encoded",
        "Next_Grade",
    ]

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df[final_cols].to_csv(PROCESSED_PATH / "clean_pgi_data.csv", index=False)
    district_map_df.to_csv(PROCESSED_PATH / "district_encoding_map.csv", index=False)

    print("Final dataset saved successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
