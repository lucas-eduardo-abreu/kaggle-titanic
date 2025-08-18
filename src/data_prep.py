# -*- coding: utf-8 -*-
"""
Módulo de preparação de dados do Titanic (versão aprimorada).

Melhorias:
- Cria novas features: FarePerPerson, TicketGroupSize, TicketFreq,
  AgeBin, Child, Mother, IsMarried, interações simples.
- Normaliza títulos e extrai Deck da cabine.

Observação:
- As regras Mother/Child são heurísticas conhecidas que ajudam em modelos lineares.
"""

from __future__ import annotations
import re
import hashlib
import pandas as pd


def _extract_title(name: str) -> str:
    """Extrai e normaliza o título do 'Name'."""
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    title = m.group(1).strip() if m else "Unknown"
    mapping = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
        "Don": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer",
        "Rev": "Officer"
    }
    return mapping.get(title, title)


def _extract_cabin_deck(cabin: str) -> str:
    """Extrai a letra do deck a partir de 'Cabin'."""
    if isinstance(cabin, str) and len(cabin) > 0:
        return cabin[0]
    return "Unknown"


def _ticket_clean(ticket: str) -> str:
    """
    Limpa formato do Ticket para reduzir cardinalidade de ruído.
    Mantém prefixos/letras quando existirem; remove pontos e barras; tira espaços extras.
    """
    if not isinstance(ticket, str):
        return "UNKNOWN"
    t = ticket.replace(".", "").replace("/", "").strip().upper()
    t = re.sub(r"\s+", "", t)
    return t if t else "UNKNOWN"


def _hash_small(text: str) -> int:
    """Hash estável pequeno para gerar keys categóricas controladas (se necessário)."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 6)


def build_features(df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Constrói o DataFrame de features X e a série alvo y (se treino).

    Novas features criadas:
    - FamilySize, IsAlone
    - Title, CabinDeck
    - FarePerPerson = Fare / FamilySize
    - TicketGroupSize = nº de passageiros com o mesmo Ticket
    - TicketFreq = frequência daquele Ticket no dataset
    - AgeBin (faixas etárias), Child (Age<16), IsMarried (Name contém 'Mrs')
    - Mother (mulher adulta com filhos e não 'Miss')
    - Interações simples: Sex_x_Pclass, IsAlone_x_Pclass
    """
    df = df.copy()

    # Alvo (se treino)
    y = df["Survived"] if (is_train and "Survived" in df.columns) else None

    # Família
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Título e Deck
    df["Title"] = df["Name"].apply(_extract_title)
    df["CabinDeck"] = df["Cabin"].apply(_extract_cabin_deck)

    # Limpeza do Ticket e estatísticas de grupo
    df["TicketClean"] = df["Ticket"].apply(_ticket_clean)
    ticket_counts = df["TicketClean"].value_counts()
    df["TicketGroupSize"] = df["TicketClean"].map(ticket_counts).fillna(1).astype(int)
    df["TicketFreq"] = df["TicketClean"].map(ticket_counts / len(df)).fillna(0.0)

    # Fare por pessoa (evita dividir por zero)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FarePerPerson"] = (df["Fare"] / df["FamilySize"].clip(lower=1)).astype(float)

    # Faixas etárias e regras Child/Mother/IsMarried
    # Age será imputada depois no pipeline, mas bins categóricos ajudam o modelo linear
    df["AgeBin"] = pd.cut(df["Age"], bins=[-1, 4, 12, 18, 30, 45, 60, 80], labels=[
        "Toddler", "Child", "Teen", "YoungAdult", "Adult", "MidAge", "Senior"
    ])
    df["Child"] = ((df["Age"].fillna(-1) < 16) & (df["Sex"] == "male")).astype(int)
    df["IsMarried"] = df["Name"].str.contains(r"\bMrs\b", regex=True).astype(int)
    df["Mother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"].fillna(99) > 18) & (df["Title"] != "Miss")).astype(int)

    # Interações simples úteis para modelos lineares
    df["Sex_x_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df["IsAlone_x_Pclass"] = df["IsAlone"].astype(str) + "_" + df["Pclass"].astype(str)

    # Seleção de colunas finais
    feature_cols = [
        # Numéricas
        "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson",
        "TicketGroupSize", "TicketFreq",
        # Categóricas
        "Pclass", "Sex", "Embarked", "Title", "CabinDeck", "AgeBin",
        "IsAlone", "Child", "IsMarried", "Mother",
        "Sex_x_Pclass", "IsAlone_x_Pclass",
    ]
    X = df[feature_cols].copy()
    return X, y
