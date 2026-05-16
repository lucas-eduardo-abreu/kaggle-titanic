# kaggle-titanic

Titanic survival prediction for the [Kaggle competition](https://www.kaggle.com/c/titanic).
Logistic Regression pipeline with feature engineering, stratified cross-validation,
and optional GridSearchCV hyperparameter tuning.

## Stack

- Python 3.13+ · pandas · scikit-learn · loguru · joblib

## Pipeline

Feature engineering: passenger title extraction (with royalty/officer normalization),
family size, alone flag, cabin deck, ticket group size, fare per person, age bins,
and interaction features (sex × class, alone × class).

`ColumnTransformer` with median imputation + scaling for numerics and most-frequent
imputation + OHE for categoricals. Classifier: `LogisticRegression(max_iter=2000)`.

## Running

```bash
pip install -r requirements.txt

# Option A — baseline (no grid search)
python train_logreg.py --train data/train.csv --out artifacts/model_logreg.joblib

# Option B — with GridSearchCV
python train_logreg_grid.py --train data/train.csv --out artifacts/model_logreg.joblib --coef-report artifacts/coef_report.csv

# Generate submission
python predict_logreg.py --model artifacts/model_logreg.joblib --test data/test.csv --out artifacts/submission.csv
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/titanic/data)
and place them in `data/`.

## Tests

```bash
pytest tests/
```

The end-to-end test trains on `data/train.csv` and asserts CV mean accuracy ≥ 75%.
