# -*- coding: utf-8 -*-
"""
Modelo de Regressão Logística com GridSearchCV opcional.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from loguru import logger
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Pré-processador: imputação, scaler e one-hot."""
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
    """Cria Pipeline base com LogisticRegression (parâmetros default bons)."""
    numeric_features = [
        "Age", "SibSp", "Parch", "Fare", "FamilySize",
        "FarePerPerson", "TicketGroupSize", "TicketFreq"
    ]
    categorical_features = [
        "Pclass", "Sex", "Embarked", "Title", "CabinDeck", "AgeBin",
        "IsAlone", "Child", "IsMarried", "Mother",
        "Sex_x_Pclass", "IsAlone_x_Pclass",
    ]

    pre = build_preprocessor(numeric_features, categorical_features)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("clf", clf)
    ])
    return pipe


def train_with_cv(X, y, cv_splits: int = 5, random_state: int = 42, do_grid: bool = True) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Treina usando CV; opcionalmente realiza GridSearchCV para ajustar hiperparâmetros.

    Retorna:
        (pipeline_ajustado, métricas)
    """
    logger.info("Iniciando treino com validação cruzada...")
    base_pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if do_grid:
        logger.info("Executando GridSearchCV para LogisticRegression...")
        param_grid = {
            "clf__solver": ["liblinear", "lbfgs", "saga"],
            "clf__penalty": ["l2"],  # l1 compatível só com liblinear/saga; mantemos l2 estável
            "clf__C": [0.5, 1.0, 1.5, 2.0, 3.0],
            # class_weight pode ajudar em leves desbalanceamentos
            "clf__class_weight": [None, "balanced"],
        }
        # Nota: 'saga' aceita l1, mas em conjunto com OHE e muitas dummies pode ficar mais lento.
        gs = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(X, y)
        pipe = gs.best_estimator_
        best_params = gs.best_params_
        best_cv = gs.best_score_
        logger.info(f"Melhor combinação: {best_params} | mean_acc={best_cv:.4f}")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=None)
    else:
        # Sem grid: avalia e ajusta
        scores = cross_val_score(base_pipe, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        base_pipe.fit(X, y)
        pipe = base_pipe
        best_params = None

    logger.info(f"Acurácias CV (refit): {scores} | média={scores.mean():.4f} ± {scores.std():.4f}")

    metrics = {
        "cv_scores": scores.tolist(),
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "best_params": best_params,
    }
    return pipe, metrics


def save_model(pipeline: Pipeline, out_path: str) -> None:
    """Salva pipeline treinado."""
    dump(pipeline, out_path)
