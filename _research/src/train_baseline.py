# -*-coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseLineTrainer:
    """
    Класс для обучения и оценки Baseline-модели (TF-IDF + Logistic Regression)
    с интеграцией мета-признаков EDA
    """
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words='english'
        )
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        self.meta_features = ['char_len', 'word_len', 'lexical_diversity', 'special_chars_ratio']

    def _prepare_features(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        """
        Векторизует текст и объединяет его с числовыми признаками EDA
        """
        texts = df['user_prompt'].fillna("").astype(str)

        if is_train:
            X_text = self.vectorizer.fit_transform(texts)
        else:
            X_text = self.vectorizer.transform(texts)

        X_meta = df[self.meta_features].values
        X_combined = hstack((X_text, X_meta))

        return X_combined


    def train(self, df_train: pd.DataFrame, target_col: str = 'jailbreak'):
        """
        Обучение модели на тренировочной выборке
        """
        logger.info(f"Подготовка признаков для Train ({df_train.shape[0]} строк)...")
        X_train = self._prepare_features(df_train, is_train=True)
        y_train = df_train[target_col].values

        logger.info("Запуск обучения Logistic Regression Baseline...")
        self.model.fit(X_train, y_train)
        logger.info("Обучение успешно завершено")

    def evaluate(self, df_eval: pd.DataFrame, name: str = "Validation", target_col: str = 'jailbreak'):
        """
        Оценка модели с выводом метрик
        """
        logger.info(f"Оценка модели на выборке: {name}")
        X_eval = self._prepare_features(df_eval, is_train=False)
        y_eval = df_eval[target_col].values

        # Получение предсказания классов и вероятностей
        preds = self.model.predict(X_eval)
        probs = self.model.predict_proba(X_eval)[:, 1]

        # Расчёт метрик
        try:
            roc_auc = roc_auc_score(y_eval, probs)
            roc_auc_str = f"{roc_auc:.4f}"
        except ValueError:
            roc_auc_str = "undefined (only one class present in y_true)"

        report = classification_report(y_eval, preds, digits=4, zero_division=0)

        print("")
        print(f"=== Результаты для выборки: {name} ===")
        print(f"ROC AUC Score: {roc_auc_str}")
        print("Classification Report:")
        print(report)