import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def titanic_df():
    rng = np.random.default_rng(42)
    n = 100
    sexes = rng.choice(["male", "female"], n)
    pclasses = rng.choice([1, 2, 3], n)
    ages = rng.uniform(1, 70, n)
    ages[rng.integers(0, n, 15)] = np.nan
    sibsp = rng.choice([0, 1, 2], n)
    parch = rng.choice([0, 1, 2], n)
    fares = rng.uniform(5, 200, n)
    titles = rng.choice(
        ["Mr", "Mrs", "Miss", "Master", "Dr", "Col", "Lady"],
        n,
    )
    names = [f"Smith, {t}. John" for t in titles]
    cabins = [
        rng.choice(["A12", "B34", "C56", np.nan])
        for _ in range(n)
    ]
    embarked = rng.choice(["S", "C", "Q", None], n)

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": rng.integers(0, 2, n),
        "Pclass": pclasses,
        "Name": names,
        "Sex": sexes,
        "Age": ages,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": [f"PC {rng.integers(10000, 99999)}" for _ in range(n)],
        "Fare": fares,
        "Cabin": cabins,
        "Embarked": embarked,
    })
