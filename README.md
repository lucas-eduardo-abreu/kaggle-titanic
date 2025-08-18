# 🚢 Titanic Survivors Challenge – Kaggle Edition

![Titanic](https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg)

> “Even God himself could not sink this ship.” – eles disseram…  
> Pois bem, vamos ver se sua **Regressão Logística** consegue prever quem afunda e quem sobrevive! ⚓😅

---

## ✨ Badges

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![Status](https://img.shields.io/badge/Leaderboard-Climb%20⛰️-brightgreen)

---

## 🎯 Objetivo

- Usar **Machine Learning** para prever quem sobreviveu ao Titanic.  
- Treinar modelos (começamos com **Regressão Logística**).  
- Gerar `submission.csv` para enviar no [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/submit).

---

## 🛠️ Estrutura do Projeto
```bash
├── data/
│ ├── train.csv # Dados de treino
│ └── test.csv # Dados de teste
├── src/
│ ├── init.py
│ ├── data_prep.py # Criação e limpeza de features
│ ├── model_logreg.py # Pipeline + GridSearchCV
│ └── inference.py # Inferência e submissão
├── artifacts/ # Modelos, métricas, submissões
├── train_logreg.py # Treino baseline
├── train_logreg_grid.py# Treino com GridSearchCV
├── predict_logreg.py # Geração de submission.csv
└── README.md
```

---

## 🔄 Fluxo do Pipeline
```mermaid
flowchart TD
    A[📂 train.csv] --> B[🔧 Data Prep<br/> (data_prep.py)]
    B --> C[🤖 Modelo<br/> Logistic Regression + GridSearchCV]
    C --> D[💾 artifacts/model.joblib]
    D --> E[📂 test.csv]
    E --> F[🔮 Inferência<br/> (predict_logreg.py)]
    F --> G[📄 submission.csv]
    G --> H[🌐 Kaggle Leaderboard]
```

---

## ⚙️ Instalação

### 1. Clone o repositório
```bash
git clone git@github.com:Baihanu/kaggle-titanic.git
cd kaggle-titanic
```

### 2. Crie o ambiente virtual
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

---

## 🚂 Treinando o modelo

### Opção A – Treino rápido (baseline)
```bash
python train_logreg.py --train data/train.csv --out artifacts/model_logreg.joblib --metrics artifacts/metrics.json
```

### Opção B – Treino com GridSearchCV
```bash
python train_logreg_grid.py --train data/train.csv --out artifacts/model_logreg.joblib --metrics artifacts/metrics.json --coef-report artifacts/coef_report.csv
```

📊 Resultados:
 - artifacts/metrics.json → métricas de CV
 - artifacts/coef_report.csv → coeficientes do modelo

---

## 🔮 Gerando a submissão
```bash
python predict_logreg.py --model artifacts/model_logreg.joblib --test data/test.csv --out artifacts/submission.csv
```
🎉 Agora você tem artifacts/submission.csv, pronto para enviar ao Kaggle!

---

## 🌍 Compatibilidade Linux/Windows
 - Scripts testados em Linux 🐧 e Windows PowerShell 🪟.
 - Ajuste apenas os separadores (/ vs. \).
 - Exemplo no Windows:
 ```bash
python .\train_logreg_grid.py --train data\train.csv --out artifacts\model_logreg.joblib --metrics artifacts\metrics.json --coef-report artifacts\coef_report.csv
```

---

## 🏆 Submissão no Kaggle
 1. Vá até a [competição no Kaggle](https://www.kaggle.com/c/titanic/submit).
 2. Clique em Submit Predictions.
 3. Envie submission.csv.
 4. Veja sua pontuação aparecer na Leaderboard. 🚀

--- 

## 😎 Dicas para melhorar a pontuação
 - Feature Engineering: idade em faixas, família, títulos (Mr, Mrs, Miss), porto de embarque.
 - Experimente Random Forest, Gradient Boosting, XGBoost, LightGBM.
 - Ajuste o threshold da Regressão Logística (não precisa ser sempre 0.5).
 - Use cross-validation para validar mudanças.

---

## 🤝 Contribuindo
Pull Requests são bem-vindos!
Sugestões de novas features, melhorias no pipeline ou até piadas com o Titanic são aceitas. 😄

---

## ⚓ Divirta-se!
<b>Spoiler histórico:</b> Jack cabia na porta sim, mas nosso modelo ainda não sabe disso... 🚪😅

