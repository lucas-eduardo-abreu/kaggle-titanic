# -*- coding: utf-8 -*-
"""
Treinamento com GridSearchCV + relatório de coeficientes.

Uso:
    python train_logreg_grid.py \
      --train data/train.csv \
      --out artifacts/model_logreg.joblib \
      --metrics artifacts/metrics.json \
      --coef-report artifacts/coef_report.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from src.data_prep import build_features
from src.model_logreg import train_with_cv, save_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="data/train.csv")
    p.add_argument("--out", type=str, default="artifacts/model_logreg.joblib")
    p.add_argument("--metrics", type=str, default="artifacts/metrics.json")
    p.add_argument("--coef-report", type=str, default="artifacts/coef_report.csv")
    return p.parse_args()


def extract_feature_names(pipeline) -> list[str]:
    """Extrai nomes das features após o preprocess (útil para coeficientes)."""
    pre = pipeline.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out().tolist()
    return feature_names


def save_coef_report(pipeline, path_out: str):
    """
    Salva um CSV com as features ordenadas por magnitude do coeficiente,
    ajudando a entender o que mais pesa na decisão.
    """
    if "clf" not in pipeline.named_steps:
        return
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return

    feature_names = extract_feature_names(pipeline)
    coefs = clf.coef_[0]  # binário
    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    out = Path(path_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_coef.to_csv(out, index=False)


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.coef_report).parent.mkdir(parents=True, exist_ok=True)

    logger.add("artifacts/train_grid.log", rotation="1 MB")
    logger.info("Carregando dados...")
    df_train = pd.read_csv(args.train)

    logger.info("Construindo features (versão aprimorada)...")
    X, y = build_features(df_train, is_train=True)

    logger.info("Treinando com GridSearchCV...")
    model, metrics = train_with_cv(X, y, cv_splits=5, random_state=42, do_grid=True)

    logger.info(f"Métricas: {metrics}")
    save_model(model, args.out)

    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_coef_report(model, args.coef_report)
    logger.success("Treino com grid concluído.")


if __name__ == "__main__":
    main()
