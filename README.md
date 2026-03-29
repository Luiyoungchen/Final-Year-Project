# District PGI Future Prediction Model – Complete Explanation

This project builds a machine learning system to predict the future performance of districts based on historical PGI (Performance Grading Index) data. The goal is to analyze how districts have performed in previous years and use that information to forecast how they will perform in the next year.

The dataset consists of multiple CSV files, each representing district-level performance for different years (2018 to 2023). These files include features such as Overall score, Outcome, ECT, IFSE, SSCP, DL, GP, Grade, and District name.

The first step in the pipeline is data preprocessing. All raw CSV files are read and standardized into a single consistent format. Since column names differ slightly across files, they are normalized and mapped correctly. The year is extracted from the file name and added as a column. All datasets are then merged into a single combined dataset.

Next, the grade values (which are categorical labels such as Prachesta, Utkarsh, etc.) are converted into numerical form so that the model can process them. District names are also encoded into numeric values using label encoding.

After this, the most important part of the project is feature engineering using lag features. Lag means using past values. Instead of using current year data to predict the same year’s grade (which causes data leakage), we use past data to predict future outcomes.

Lag1 represents the previous year’s value, and lag2 represents the value from two years before. For example, to predict the grade of 2024, the model uses data from 2022 and 2023. This allows the model to understand patterns such as improvement, decline, or stability over time.

Additional features are created from lag values, such as trend (difference between lag1 and lag2), average performance, and variation. These features help the model capture behavior patterns rather than just raw values.

A target variable called Next_Grade is created by shifting the grade column forward by one year. This represents the future grade that the model needs to predict. Rows that do not have enough past data (for lag features) or future data (for Next_Grade) are removed. This is why the dataset size reduces after preprocessing. However, this is necessary to ensure that every training example has complete information.

Initially, the model was trained to predict exact grades (1 to 6), but this resulted in lower accuracy due to noise and complexity. To improve performance, grades were grouped into three categories:
Low (1–2), Medium (3–4), and High (5–6). This simplified the problem and improved model accuracy to over 60%.

The model is then trained using algorithms such as Random Forest or Gradient Boosting. The training process involves learning patterns from historical data, where past performance (lag features) is used to predict future grade categories.

Once the model is trained, it can be used for future prediction. For predicting 2024, the system takes the latest available data (2022 and 2023), creates lag features, and feeds them into the trained model. Since future data is not available during prediction, no rows are removed in this phase.

It is important to understand that removing the last year’s data happens only during training because the model requires known future values to learn. During prediction, the model uses the latest available data and generates the future prediction on its own.

The final output of the system is a prediction of the district’s performance category (Low, Medium, or High) for the next year. This can be used for decision-making, identifying improving or declining districts, and supporting data-driven educational planning.

Overall, this project demonstrates a complete machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and real-world future prediction.



# 🚀 Developer Guide: How to Run and Continue the PGI Prediction Model

## 1. Project Setup

1. Clone or download the project folder.
2. Ensure Python (3.9+) is installed.
3. Install required libraries:

pip install pandas scikit-learn joblib

---

## 2. Folder Structure (Important)

Make sure the project follows this structure:

"pgi/
│
├── data/
│   ├── raw/                # Place all raw CSV files (2018–2023)
│   └── processed/          # Will be auto-created
│
├── models/
│   └── (model will be saved here)
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict_2024.py"

---

## 3. Step 1: Add Raw Data

* Place all district PGI CSV files inside:

data/raw/

Example:

* District_PGI_2018_19.csv
* District_PGI_2019_20.csv
* ...
* District_PGI_2023_24.csv

---

## 4. Step 2: Run Preprocessing

Command:

python src/preprocess.py

What it does:

* Reads all raw files
* Standardizes columns
* Encodes district names
* Converts grades to numbers
* Creates lag features (lag1, lag2)
* Creates target variable (Next_Grade)
* Creates Grade_Group (Low/Medium/High)
* Removes invalid rows
* Saves clean dataset to:

data/processed/clean_pgi_data.csv

Also saves:
data/processed/district_encoding_map.csv

---

## 5. Step 3: Train the Model

Command:

python src/train.py

What it does:

* Loads processed dataset
* Trains ML model (Random Forest / Boosting)
* Uses lag-based features
* Predicts Grade_Group
* Evaluates accuracy (~60%+)
* Saves model to:

models/pgi_model.pkl

---

## 6. Step 4: Predict Future (2024)

Command:

python src/predict_2024.py

What it does:

* Loads trained model
* Uses latest available data (2022 & 2023)
* Creates lag features
* Predicts 2024 performance category
* Saves output to:

data/processed/predictions_2024.csv

Output example:
District | Year | Predicted_Label
Agra     | 2024 | Medium

---

## 7. Backend Integration (Next Step)

To integrate with backend:

1. Load model using joblib
2. Accept district name as input
3. Fetch last 2 years of that district
4. Create lag features
5. Call model.predict()
6. Return result

Example flow:

Input → District Name
↓
Fetch data (2022, 2023)
↓
Create features (lag1, lag2, trend)
↓
Model prediction
↓
Return "Low / Medium / High"

---

## 8. Important Notes

* Do NOT modify training logic unless needed
* Do NOT use current year data for prediction (avoid leakage)
* Last year is removed only during training (this is correct)
* Prediction uses latest available data (no removal)

---

## 9. If Adding New Data (Future)

If new year data (e.g., 2024 actual) is added:

1. Place new CSV in data/raw/
2. Re-run preprocess.py
3. Re-run train.py

Model will automatically improve with more data

---

## 10. Summary

Pipeline:

Raw Data → Preprocess → Train → Predict

The system predicts future district performance using past trends.
