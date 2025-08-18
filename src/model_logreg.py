# -*- coding: utf-8 -*-
"""
Módulo do modelo de Regressão Logística para o Titanic.

Cria um Pipeline com:
- Pré-processamento (imputação, padronização e one-hot)
- Classificador LogisticRegression

Inclui função de treino com validação cruzada para aferir desempenho.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from loguru import logger
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Monta o pré-processador com imputação, scaler e one-hot encoding."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre


def build_pipeline() -> Pipeline:
    """
    Cria o Pipeline final com pré-processamento e Regressão Logística.

    Retorna:
        sklearn.Pipeline: pipeline pronto para ajuste.
    """
    numeric_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    categorical_features = ["Pclass", "Sex", "Embarked", "IsAlone", "Title", "CabinDeck"]

    pre = build_preprocessor(numeric_features, categorical_features)

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("clf", clf)
    ])
    return pipe


def train_with_cv(X, y, cv_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Treina o pipeline usando validação cruzada estratificada.

    Retorna:
        (pipeline_ajustado, métricas) onde métricas contém mean_acc, std_acc e scores.
    """
    logger.info("Iniciando treino com validação cruzada...")
    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=None)

    logger.info(f"Acurácias CV: {scores} | média={scores.mean():.4f} ± {scores.std():.4f}")

    # Ajusta no dataset completo após CV para salvar modelo final
    pipe.fit(X, y)

    metrics = {
        "cv_scores": scores.tolist(),
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std())
    }
    return pipe, metrics


def save_model(pipeline: Pipeline, out_path: str) -> None:
    """Salva o pipeline treinado com joblib."""
    dump(pipeline, out_path)
