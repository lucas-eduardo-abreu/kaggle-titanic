# sanity_check.py (corrigido)
# -*- coding: utf-8 -*-
import pandas as pd
from src.data_prep import build_features
from src.model_logreg import build_pipeline

print("Lendo data/train.csv ...")
df = pd.read_csv("data/train.csv")
X, y = build_features(df, is_train=True)
print("Shapes:", X.shape, y.shape)

pipe = build_pipeline()
print("Rodando fit rápido...")
pipe.fit(X, y)  # sem fillna(0)
print("OK, pipeline ajustou sem erros.")
