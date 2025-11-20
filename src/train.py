import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import shap
import optuna
import warnings
warnings.filterwarnings('ignore')

from preprocess import build_preprocessor
from modeling import get_base_pipelines, calibrate_model
import logging
logging.basicConfig(
    leve=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

DATA_PATH = '../data/raw/Human_Resources.csv'
TARGET = 'Attrition'
RESULT_DIR = '../models'
os.makedirs(RESULT_DIR, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLIT = 4
N_ITER_RANDOM = 30

def load_data(path):
    df = pd.read_csv(path)
    return df

def main():
    df = load_data(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f'{TARGET} not found in data.')
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,stratify=y
    )
    logging.info(f'Train: {X_train.shape}, Holdout: {X_holdout.shape}')

    preprocessor, metadata = build_preprocessor(X_train)
    logging.info(f'Preprocessor built. Num cols: {len(metadata['num_cols'])}, Cat cols: {len(metadata['cat_cols'])}')

    base_pipes = get_base_pipelines(preprocessor)


if __name__ == '__main__':
    main()



