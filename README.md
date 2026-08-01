# California Housing Price Predictor

Predicts median house value for California census block groups from the classic
**California Housing** dataset, using a Random Forest Regressor trained with scikit-learn.

## Overview

The dataset contains ~20,600 records at the census block-group level, describing location
(longitude, latitude), housing characteristics (age, room/bedroom counts, population,
households), median income, and proximity to the ocean.

The target, `median_house_value`, is a continuous dollar value.

## Pipeline

1. **Exploratory data analysis**: distribution plots, geographic scatter plots colored by
   price, and a correlation matrix to identify `median_income` as the strongest single
   predictor of house value.
2. **Stratified train/test split** (80/20) on an income bucket derived from `median_income`,
   so both splits are representative of the full income distribution.
3. **Preprocessing** via a scikit-learn `ColumnTransformer`:
   - Numeric features → median imputation + standard scaling
   - `ocean_proximity` (categorical) → one-hot encoding
4. **Model**: `RandomForestRegressor`.
5. **Evaluation**: 10-fold cross-validation on the training set, plus a final check on a
   held-out test set the model never saw during training.

## Results

Evaluated on a held-out test set (~4,100 samples):

| Metric | Score |
|---|---|
| Cross-val RMSE | ~$49,300 |
| Test RMSE | ~$47,000 |
| Mean house value (training data) | ~$207,000 |

An RMSE of ~$47K against a mean value of ~$207K means predictions are typically within
about 23% of the true price — a reasonable baseline, with room to improve (see Notes below).

## Project structure

```
california-housing-price-predictor/
├── data/
│   └── housing.csv                      # raw data
├── models/
│   ├── model.pkl                        # trained RandomForestRegressor
│   └── pipeline.pkl                     # fitted preprocessing pipeline
├── src/
│   ├── train.py                         # trains model, saves train/test split + metrics
│   └── predict.py                       # runs inference on new data
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt

# Train the model (creates data/train.csv, data/test.csv, models/*.pkl)
python src/train.py

# Run predictions on the held-out test set (or swap in your own data)
python src/predict.py
```

## Notes / possible next steps

- Try feature engineering used in similar projects on this dataset, e.g.
  `rooms_per_household`, `bedrooms_per_room`, `population_per_household` — these
  often correlate more strongly with price than the raw counts.
- Hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV` on `n_estimators`,
  `max_features`, `max_depth`) would likely reduce RMSE further.
- Try `GradientBoostingRegressor` or `XGBoost` as stronger baselines to compare against
  the Random Forest.

## Disclaimer

This is a portfolio/learning project based on 1990 U.S. Census data and is not intended
for real-world property valuation.
