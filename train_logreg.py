# -*- coding: utf-8 -*-
"""
Script de treinamento do modelo de Regressão Logística para o Titanic.

Fluxo:
1) Carrega train.csv
2) Constrói features (DataPrep)
3) Treina com validação cruzada
4) Salva modelo e métricas

Uso:
    python train_logreg.py --train data/train.csv --out artifacts/model_logreg.joblib --metrics artifacts/metrics.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from loguru import logger
from src.data_prep import build_features
from src.model_logreg import train_with_cv, save_model


def parse_args():
    """Parsa argumentos de linha de comando."""
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="data/train.csv", help="Caminho para o train.csv")
    p.add_argument("--out", type=str, default="artifacts/model_logreg.joblib", help="Arquivo de saída do modelo")
    p.add_argument("--metrics", type=str, default="artifacts/metrics.json", help="Arquivo JSON de métricas")
    return p.parse_args()


def main():
    """Executa o pipeline de treino completo e persiste artefatos."""
    args = parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)

    logger.add("artifacts/train.log", rotation="1 MB")
    logger.info("Carregando dados de treino...")
    df_train = pd.read_csv(args.train)

    logger.info("Construindo features...")
    X, y = build_features(df_train, is_train=True)

    logger.info("Treinando modelo (CV Estratificada)...")
    model, metrics = train_with_cv(X, y, cv_splits=5, random_state=42)

    logger.info(f"Métricas: {metrics}")
    logger.info(f"Salvando modelo em: {args.out}")
    save_model(model, args.out)

    logger.info(f"Salvando métricas em: {args.metrics}")
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.success("Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()
