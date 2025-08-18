# -*- coding: utf-8 -*-
"""
Script de inferência para gerar o submission.csv do Titanic.

Fluxo:
1) Carrega o pipeline treinado
2) Constrói X_test a partir do test.csv
3) Prediz e salva o arquivo de submissão

Uso:
    python predict_logreg.py \
        --model artifacts/model_logreg.joblib \
        --test data/test.csv \
        --out artifacts/submission.csv
"""

from __future__ import annotations
import argparse
from loguru import logger
from src.inference import (
    load_trained_pipeline,
    prepare_test_features,
    predict_and_build_submission,
    save_submission_csv,
)


def parse_args():
    """Parsa argumentos de linha de comando."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="artifacts/model_logreg.joblib", help="Caminho do modelo treinado")
    p.add_argument("--test", type=str, default="data/test.csv", help="Caminho para o test.csv")
    p.add_argument("--out", type=str, default="artifacts/submission.csv", help="Arquivo de saída (submission.csv)")
    return p.parse_args()


def main():
    """Executa a geração do submission.csv."""
    args = parse_args()
    logger.add("artifacts/predict.log", rotation="1 MB")

    pipeline = load_trained_pipeline(args.model)
    X_test, passenger_ids = prepare_test_features(args.test)
    submission = predict_and_build_submission(pipeline, X_test, passenger_ids)
    save_submission_csv(submission, args.out)

    logger.success("Inference completed successfully.")


if __name__ == "__main__":
    main()
