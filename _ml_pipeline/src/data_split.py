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
    Реализует сэмплирование данных для MVP и выделение сэмпла для стресс-теста (длинные нейстральные тексты).
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
            micro_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Разделяет датафрейм на выборки: Train, Micro-Train, Val, Test и Stress-Test.
        """
        logger.info("Запуск процесса стратегического сплитования данных...")
        df_clean = df.copy()

        # 1. Находим кандидатов для стресс-теста (длинные нейтральные тексты)
        stress_mask = (df_clean[self.target_col] == 0) & (df_clean[self.word_len_col] > self.stress_test_threshold)
        df_stress_candidate = df_clean[stress_mask]
        df_pool = df_clean[~stress_mask]

        logger.info(f"Найдено кандидатов (длинных не-атак): {df_stress_candidate.shape[0]}")

        # Проверка безопасности: если кандидаты есть, берем 10% в Stress-Test, а 90% возвращаем в пул обучения
        if not df_stress_candidate.empty:
            df_stress_test, df_long_normal_train = train_test_split(
                df_stress_candidate,
                test_size=0.9,  # 90% возвращаем в общий пул
                random_state=self.random_state
            )
            df_pool = pd.concat([df_pool, df_long_normal_train], ignore_index=True)
            logger.info(f"Выделено в финальный Stress-Test (10% от кандидатов): {df_stress_test.shape[0]}")
            logger.info(f"Возвращено в обучающий пул для репрезентативности (90%): {df_long_normal_train.shape[0]}")
        else:
            logger.warning("Кандидаты для стресс-теста не найдены. Стресс-тест будет пустым.")
            df_stress_test = pd.DataFrame(columns=df_clean.columns)

        # Проверка на наличие данных для дальнейшего сплита
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

        # 3. Формирование Micro-Train сэмпла для быстрой проверки MVP
        _, df_train_micro = train_test_split(
            df_train,
            test_size=micro_size,
            stratify=df_train[self.target_col],
            random_state=self.random_state
        )

        logger.info("Сплитование успешно завершено")
        self._print_stats(df_train, df_train_micro, df_val, df_test, df_stress_test)

        return {
            'train': df_train,
            'train_micro': df_train_micro,
            'val': df_val,
            'test': df_test,
            'stress_test': df_stress_test
        }

    def _print_stats(self, train: pd.DataFrame, micro: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, stress: pd.DataFrame):
        """
        Вывод баланса классов по выборкам для контроля утечек
        """
        for name, sdf in zip(['Train', 'Micro-Train', 'Validation', 'Test', 'Stress-Test'], [train, micro, val, test, stress]):
            total = sdf.shape[0]
            if total > 0 and self.target_col in sdf.columns:
                pos_ratio = (sdf[self.target_col] == 1).sum() / total * 100
                print(f"-> {name}: {total} строк, атак (класс 1): {pos_ratio:.2f}%")
            else:
                print(f"-> {name}: {total} строк (стратификация не применима / нет атак)")