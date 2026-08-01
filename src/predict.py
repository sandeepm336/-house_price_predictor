"""
Run inference on new data using the trained model + pipeline.

Run:
    python src/predict.py

Reads data/test.csv (or any file with the same columns, minus median_house_value)
and writes predictions to data/predictions.csv
"""

import joblib
import pandas as pd

MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"
INPUT_FILE = "data/test.csv"
OUTPUT_FILE = "data/predictions.csv"


def main():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_FILE)
    features = input_data.drop(columns=["median_house_value"], errors="ignore")

    transformed = pipeline.transform(features)
    input_data["predicted_median_house_value"] = model.predict(transformed).round(0)

    input_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
