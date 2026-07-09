# -*- coding: utf-8 -*-
import re
import pandas as pd


class TextFeatureExtractor:
    """
    Класс для детерменированного feature engineering.
    Отделяет логику генерации признаков от основного пайплайна данных.
    """
    def __init__(self, text_column: str = 'user_prompt'):
        self.text_column = text_column
        self.spec_char_regex = re.compile(r'[^a-zA-Z0-9\s]')

    def _extract_row_features(self, text: str) -> dict:
        # Проверка на пустые строки или некорректный тип данных
        if not isinstance(text, str) or not text.strip():
            return {
                'char_len': 0,
                'word_len': 0,
                'lexical_diversity': 0.0,
                'special_chars_ratio': 0.0
            }

        char_len = len(text)
        words = text.split()
        word_len = len(words)

        # Вычисление Lexical Diversity: удельная составляющая уникальных слов
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_len if word_len > 0 else 0.0

        # Вычисление количества спецсимволов
        special_chars = len(self.spec_char_regex.findall(text))
        special_char_ratio = special_chars / char_len if char_len > 0 else 0.0

        return {
            'char_len': char_len,
            'word_len': word_len,
            'lexical_diversity': lexical_diversity,
            'special_chars_ratio': special_char_ratio
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает фичи из датафрейма и возвращает обогазённый датафрейм.
        """
        df_out = df.copy()

        features_list = df_out[self.text_column].apply(self._extract_row_features).tolist()
        features_df = pd.DataFrame(features_list, index=df_out.index)

        return pd.concat([df_out, features_df], axis=1)

