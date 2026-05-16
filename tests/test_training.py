import json
import importlib
from pathlib import Path
import pytest
from src.data_prep import build_features
from src.model_logreg import train_with_cv, save_model


MIN_MEAN_ACC = 0.75


def test_data_prep_importable():
    """Importing data_prep must not raise SyntaxError or NameError."""
    import src.data_prep as dp
    assert callable(dp.build_features)


def test_train_logreg_end_to_end(tmp_path: Path, titanic_df):
    X, y = build_features(titanic_df, is_train=True)

    model, metrics = train_with_cv(X, y, cv_splits=3, random_state=42)

    assert "mean_accuracy" in metrics
    assert metrics["mean_accuracy"] >= 0.5  # lenient threshold for 100-row synthetic data

    model_path = tmp_path / "model_logreg.joblib"
    save_model(model, str(model_path))
    assert model_path.exists()

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    assert metrics_path.exists()


def test_build_features_shape(titanic_df):
    X, y = build_features(titanic_df, is_train=True)
    assert len(X) == len(titanic_df)
    assert y is not None
    assert set(y.unique()).issubset({0, 1})


def test_build_features_no_train_target(titanic_df):
    df_no_survived = titanic_df.drop(columns=["Survived"])
    X, y = build_features(df_no_survived, is_train=False)
    assert y is None
    assert len(X) == len(df_no_survived)
