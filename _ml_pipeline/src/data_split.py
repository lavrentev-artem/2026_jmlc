# -*- coding: utf-8 -*-
import logging
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataStratifier:
    """
    Класс для стратегического сплитования данных.
    Реализует выделение сэмпла для стресс-теста (длинные нейтральные тексты)
    и пропорциональное уменьшение датасета для отладочных выборок.
    """
    def __init__(
            self,
            target_col: str = 'jailbreak',
            word_len_col: str = 'word_len',
            stress_test_threshold: int = 120,
            random_state: int = 42
    ):
        self.target_col = target_col
        self.word_len_col = word_len_col
        self.stress_test_threshold = stress_test_threshold
        self.random_state = random_state

    def split_data(
            self,
            df: pd.DataFrame,
            train_size: float = 0.8,
            fraction: float = 1.0  # Доля от всего датасета, которую нужно использовать (1.0 - full, 0.1 - micro, 0.01 - nano)
    ) -> Dict[str, pd.DataFrame]:
        """
        Разделяет датафрейм на выборки: Train, Val, Test и Stress-Test с сохранением пропорций.
        """
        logger.info(f"Запуск стратегического сплитования (Коэффициент сжатия: {fraction})...")
        df_clean = df.copy()

        # 0. Если требуется уменьшенная выборка (micro/nano) - честно стратифицированно сжимаем весь исходный датасет
        if fraction < 1.0:
            _, df_clean = train_test_split(
                df_clean,
                test_size=fraction,
                stratify=df_clean[self.target_col],
                random_state=self.random_state
            )

        # 1. Находим кандидатов для стресс-теста в текущем объеме данных
        stress_mask = (df_clean[self.target_col] == 0) & (df_clean[self.word_len_col] > self.stress_test_threshold)
        df_stress_candidate = df_clean[stress_mask]
        df_pool = df_clean[~stress_mask]

        # Проверка безопасности: берем 10% в Stress-Test, 90% возвращаем в пул
        if not df_stress_candidate.empty:
            # Для очень маленьких выборок (nano) может оказаться всего 1-2 кандидата, тогда train_test_split упадет.
            # Защита от этого:
            if len(df_stress_candidate) > 5:
                df_stress_test, df_long_normal_train = train_test_split(
                    df_stress_candidate,
                    test_size=0.9,
                    random_state=self.random_state
                )
                df_pool = pd.concat([df_pool, df_long_normal_train], ignore_index=True)
            else:
                # Если кандидатов слишком мало, отдаем их все в пул обучения, а стресс-тест оставляем пустым
                df_pool = pd.concat([df_pool, df_stress_candidate], ignore_index=True)
                df_stress_test = pd.DataFrame(columns=df_clean.columns)
                logger.warning(f"Мало кандидатов ({len(df_stress_candidate)}) для {fraction} сплита. Все ушли в обучение.")
        else:
            df_stress_test = pd.DataFrame(columns=df_clean.columns)

        if df_pool.empty:
            raise ValueError("Пул данных пуст после извлечения Стресс-теста.")

        # 2. Выделение Train, Val, Test из общего пула
        val_test_size = 1.0 - train_size
        df_train, df_val_test = train_test_split(
            df_pool,
            test_size=val_test_size,
            stratify=df_pool[self.target_col],
            random_state=self.random_state
        )

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            stratify=df_val_test[self.target_col],
            random_state=self.random_state
        )

        # self._print_stats(df_train, df_val, df_test, df_stress_test) # Раскомментировать для отладки внутри класса

        return {
            'train': df_train,
            'val': df_val,
            'test': df_test,
            'stress_test': df_stress_test
        }

    def _print_stats(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, stress: pd.DataFrame):
        for name, sdf in zip(['Train', 'Validation', 'Test', 'Stress-Test'], [train, val, test, stress]):
            total = sdf.shape[0]
            if total > 0 and self.target_col in sdf.columns:
                pos_ratio = (sdf[self.target_col] == 1).sum() / total * 100
                print(f"-> {name}: {total} строк, атак (класс 1): {pos_ratio:.2f}%")
            else:
                print(f"-> {name}: {total} строк (стратификация не применима / нет атак)")