# -*- coding: utf-8 -*-
"""
Módulo de preparação de dados do Titanic.

Responsável por:
- Limpeza e criação de features (FamilySize, IsAlone, Title, CabinDeck)
- Separação de X (features) e y (alvo) a partir do train.csv

"""

from __future__ import annotations
import re
import pandas as pd


def _extract_title(name: str) -> str:
    """Extrai o título do passageiro a partir do campo 'Name' (ex.: Mr, Mrs, Miss)."""
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    title = m.group(1).strip() if m else "Unknown"
    # Normaliza títulos raros
    mapping = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
        "Don": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer",
        "Rev": "Officer"
    }
    return mapping.get(title, title)


def _extract_cabin_deck(cabin: str) -> str:
    """Extrai a letra do deck a partir de 'Cabin' (ex.: 'C85' -> 'C')."""
    if isinstance(cabin, str) and len(cabin) > 0:
        return cabin[0]
    return "Unknown"


def build_features(df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Constrói o DataFrame de features X e a série alvo y (se treino).

    - Cria FamilySize e IsAlone.
    - Extrai Title e CabinDeck.
    - Retorna X com colunas padronizadas e y quando disponível.

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        is_train (bool): Se True, retorna também y com a coluna 'Survived'.

    Retorna:
        (X, y): X é pd.DataFrame; y é pd.Series ou None se is_train=False.
    """
    df = df.copy()

    # Features de família
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Título e Deck
    df["Title"] = df["Name"].apply(_extract_title)
    df["CabinDeck"] = df["Cabin"].apply(_extract_cabin_deck)

    # Seleção de colunas para o modelo
    feature_cols = [
        "Pclass", "Sex", "Embarked",
        "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "Title", "CabinDeck"
    ]
    X = df[feature_cols].copy()

    y = df["Survived"] if (is_train and "Survived" in df.columns) else None
    return X, y
