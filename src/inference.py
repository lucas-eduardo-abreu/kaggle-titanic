# -*- coding: utf-8 -*-
"""
Módulo de inferência para o desafio Titanic.

Responsável por:
- Carregar o pipeline treinado (joblib)
- Ler o test.csv, construir features com o mesmo data prep do treino
- Gerar previsões e montar o submission.csv (PassengerId, Survived)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from joblib import load
from loguru import logger
from .data_prep import build_features


def load_trained_pipeline(model_path: str):
    """Carrega o pipeline treinado salvo via joblib."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info(f"Loading trained pipeline from: {model_path}")
    pipeline = load(model_path)
    return pipeline


def prepare_test_features(test_csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega o test.csv e constrói as features X_test usando o mesmo processo do treino.

    Retorna:
        (X_test, passenger_ids)
    """
    logger.info(f"Reading test data from: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)

    if "PassengerId" not in df_test.columns:
        raise ValueError("Column 'PassengerId' not found in test.csv.")

    # is_train=False para não tentar extrair y
    X_test, _ = build_features(df_test, is_train=False)
    passenger_ids = df_test["PassengerId"]
    return X_test, passenger_ids


def predict_and_build_submission(pipeline, X_test: pd.DataFrame, passenger_ids: pd.Series) -> pd.DataFrame:
    """
    Gera as previsões (0/1) e retorna um DataFrame no formato exigido pelo Kaggle.

    Formato:
        PassengerId, Survived
    """
    logger.info("Running predictions on test set...")
    preds = pipeline.predict(X_test)

    # Garante inteiros 0/1
    preds = (pd.Series(preds).astype(int)).values

    submission = pd.DataFrame({
        "PassengerId": passenger_ids.astype(int),
        "Survived": preds
    })
    return submission


def save_submission_csv(submission: pd.DataFrame, out_csv_path: str) -> None:
    """Salva o submission.csv sem índice."""
    out = Path(out_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    logger.success(f"Submission saved to: {out}")
