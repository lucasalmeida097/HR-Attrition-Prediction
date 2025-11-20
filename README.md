# 📊 HR Attrition Prediction – Data Science & Business Intelligence

## 🔎 Contexto do Problema e Motivação de Negócios

O **turnover** (rotatividade de funcionários) é um desafio crítico e oneroso para qualquer organização. Este projeto visa quantificar e mitigar esse risco desenvolvendo um sistema de previsão e análise de *drivers*. 

* O custo de substituir um colaborador pode variar de **50% a 200% do salário anual**.
* A motivação é transformar dados brutos em **conhecimento institucional** (*institutional knowledge*) para ações estratégicas de retenção.

---

## 🎯 Objetivo e Proposta de Valor

O objetivo primário é desenvolver uma solução de **Machine Learning** que ofereça **previsibilidade** e **insights acionáveis** ao time de Recursos Humanos (RH) e à Liderança.

| Foco | Proposta de Valor |
| :--- | :--- |
| **Previsão** | Estimar a **probabilidade de saída (Attrition)** de cada colaborador. |
| **Análise de Risco** | Identificar os **principais perfis de risco** por meio de análises visuais de *features* (departamento, idade, satisfação, etc.). |
| **Ação** | Fornecer a **Feature Importance** para orientar as ações estratégicas do RH (foco em *features* de alto impacto). |
| **Estratégia** | Suporte à tomada de decisão via **dashboards** (futuro *frontend*). |

---

## 🛠️ Stack de Tecnologias e Bibliotecas

A solução é construída com a *stack* Python, focada em performance e **escalabilidade**.

| Categoria | Tecnologia / Library | Finalidade |
| :--- | :--- | :--- |
| **Data Core** | **Pandas, NumPy** | Manipulação, limpeza, análise e cálculo vetorial de dados. |
| **Machine Learning** | **Scikit-learn** | *Pipelines* de pré-processamento, modelos base e avaliação. |
| **Boosting** | **XGBoost, LightGBM, CatBoost** | Algoritmos de *Gradient Boosting* de alta performance. |
| **Balanceamento** | **Imblearn (SMOTE)** | Tratamento de desbalanceamento da classe (`Attrition`). |
| **Otimização** | **Optuna** | Framework para **Hyperparameter Optimization (HPO)**. |
| **Interpretabilidade** | **SHAP** | Explicação do impacto de cada *feature* nas previsões do modelo. |
| **Deployment** | **FastAPI, Docker** | Servir o modelo treinado como uma API e containerização da solução. |

---

## 📂 Estrutura do Repositório (Project Structure)

A arquitetura segue o princípio de separação de responsabilidades (*Separation of Concerns*) e facilita a manutenção e o *deployment*.

```
hr-attrition-prediction/
│
├── data/
│   ├── raw/                 # Dataset original (Human_Resources.csv)
│   └── processed/           # Dados limpos e prontos para modelagem
│
├── notebooks/
│   ├── 01_eda.ipynb         # Análise Exploratória de Dados (EDA)
│   └── 02_pipeline.ipynb    # Protótipos e testes de pipelines
│
├── src/                     # Código fonte modular e reutilizável
│   ├── preprocess.py        # Módulo: Pipelines de Pré-processamento (ColumnTransformer)
│   ├── modeling.py          # Módulo: Definição dos Modelos e Pipelines   Base
│   └── train.py             # Script principal de orquestração, treino e avaliação
│
├── models/                  # Diretório para modelos treinados serializados (*joblib*)
├── api/                     # Futura API (FastAPI) para servir previsões
├── frontend/                # Futura interface (Streamlit)
├── Dockerfile               # Configuração para containerização
└── requirements.txt         # Lista de dependências Python

```
---

## 📈 Metodologia de Modelagem e Avaliação

O *workflow* de modelagem é rigoroso para garantir resultados confiáveis e robustos.

### Modelos Utilizados

1.  **Regressão Logística (Logistic Regression):** Usado como **baseline** devido à sua interpretabilidade.
2.  **Random Forest (RF) / Gradient Boosting (GB):** Modelos de *ensemble* que oferecem maior poder preditivo.
3.  **SVC (Support Vector Classifier):** Incluído para exploração de hiperplanos de separação.

### Estratégia de Treinamento

* **Data Split:** Divisão rigorosa em **Training Set** e **Holdout Set** (usado apenas para avaliação final imparcial).
* **Pipeline:** Uso de `ColumnTransformer` (em `preprocess.py`) integrado a cada modelo via `Pipeline` para evitar **Data Leakage**.
* **Balanceamento:** Utilização de técnicas como `SMOTE` (via `imblearn`) e `class_weight='balanced'` para lidar com o desbalanceamento da variável **`Attrition`**.
* **Validação:** Uso de `StratifiedKFold` para garantir a representatividade das classes em todos os *folds* de validação cruzada.

### Métricas de Performance

Como o foco é identificar e intervir em casos de alto risco, as métricas são priorizadas:

* **Recall (Classe Positiva):** Crucial para garantir que o modelo consiga identificar a maior quantidade possível de colaboradores que *vão* sair (minimizar *False Negatives*).
* **F1-Score (Macro):** Métrica principal para balancear *Precision* e *Recall* no cenário desbalanceado.
* **AUC-ROC:** Medida da capacidade de discriminação do modelo.

---

## 🚀 Impacto Esperado

O sucesso deste projeto será medido pela capacidade de reduzir o **custo total do turnover** e aumentar a retenção de talentos através de decisões baseadas em dados. O resultado final será um **modelo preditivo calibrado** e um conjunto de **insights acionáveis**.
