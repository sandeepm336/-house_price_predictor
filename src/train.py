"""
Train a Random Forest Regressor to predict median house value from the
California Housing dataset (block-group level census data, 1990).

Run:
    python src/train.py

Produces:
    models/model.pkl
    models/pipeline.pkl
    data/train.csv
    data/test.csv   (held out, never seen during training)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "data/housing.csv"
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"

TARGET = "median_house_value"
CAT_ATTRIBS = ["ocean_proximity"]


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])


def main():
    data = pd.read_csv(DATA_FILE)

    # Median income is the strongest predictor of house value, so we stratify
    # the split on an income bucket to keep train/test representative.
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(data, data["income_cat"]))
    train_set = data.loc[train_idx].drop(columns=["income_cat"]).reset_index(drop=True)
    test_set = data.loc[test_idx].drop(columns=["income_cat"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    train_set.to_csv(TRAIN_FILE, index=False)
    test_set.to_csv(TEST_FILE, index=False)

    X_train = train_set.drop(columns=[TARGET])
    y_train = train_set[TARGET]

    num_attribs = X_train.drop(columns=CAT_ATTRIBS).columns.tolist()
    pipeline = build_pipeline(num_attribs, CAT_ATTRIBS)
    X_train_prepared = pipeline.fit_transform(X_train)

    model = RandomForestRegressor(
        random_state=42, n_estimators=100, max_depth=20, n_jobs=-1
    )

    # 10-fold cross-validation on the training set (honest, no leakage)
    cv_rmse = -cross_val_score(
        model, X_train_prepared, y_train,
        scoring="neg_root_mean_squared_error", cv=10,
    )
    print(f"Cross-val RMSE: ${cv_rmse.mean():,.0f} (+/- ${cv_rmse.std():,.0f})")

    model.fit(X_train_prepared, y_train)

    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"Saved {MODEL_FILE} and {PIPELINE_FILE}")

    # ---- Honest evaluation on the held-out TEST set (never used in training) ----
    X_test = test_set.drop(columns=[TARGET])
    y_test = test_set[TARGET]
    X_test_prepared = pipeline.transform(X_test)
    preds = model.predict(X_test_prepared)

    test_rmse = root_mean_squared_error(y_test, preds)
    print("\n=== Held-out test set performance ===")
    print(f"RMSE: ${test_rmse:,.0f}  (mean house value in training data: ${y_train.mean():,.0f})")


if __name__ == "__main__":
    main()
