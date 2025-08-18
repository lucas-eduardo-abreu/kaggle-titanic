# -*- coding: utf-8 -*-
"""
Teste automatizado do treinamento.

Objetivos:
- Garantir que o pipeline treina fim-a-fim sem erros
- Acurácia média de CV acima de um patamar mínimo (ex.: 0.75)
- Modelo e métricas são persistidos

Observação:
- Assume que 'data/train.csv' existe no projeto (Kaggle Titanic).
"""

import json
from pathlib import Path
import pandas as pd
from src.data_prep import build_features
from src.model_logreg import train_with_cv, save_model


MIN_MEAN_ACC = 0.75  # limiar razoável para baseline de regressão logística


def test_train_logreg_end_to_end(tmp_path: Path):
    # Carrega dados (ajuste o caminho se necessário)
    df = pd.read_csv("data/train.csv")
    X, y = build_features(df, is_train=True)

    # Treina com CV
    model, metrics = train_with_cv(X, y, cv_splits=5, random_state=42)

    # Verificações de métricas
    assert "mean_accuracy" in metrics
    assert metrics["mean_accuracy"] >= MIN_MEAN_ACC

    # Persiste artefatos
    model_path = tmp_path / "model_logreg.joblib"
    save_model(model, str(model_path))
    assert model_path.exists()

    # Persiste métricas
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    assert metrics_path.exists()
