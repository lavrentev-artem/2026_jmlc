# -*- coding: utf-8 -*-
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd

# Гарантируем, что корневая папка _ml_pipeline находится в sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.features import TextFeatureExtractor
from src.data_split import DataStratifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_split_report(full_splits: Dict[str, pd.DataFrame], micro_splits: Dict[str, pd.DataFrame], output_dir: Path,
                      target_col: str = 'jailbreak'):
    """
    Создает расширенный текстовый паспорт данных, фиксируя объемы обеих экосистем (Full и Micro).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"data_split_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт выгрузки датасета (Двухконтурный: Full & Micro)\n")
        f.write(f"Дата и время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- ОСНОВНОЙ НАБОР ДАННЫХ ---\n")
        for name, df in full_splits.items():
            total = df.shape[0]
            attack_count = int(df[target_col].sum()) if target_col in df.columns else 0
            attack_rate = (attack_count / total) * 100 if total > 0 else 0
            f.write(f"-> {name.capitalize()}: {total} строк | Атак: {attack_count} шт. ({attack_rate:.2f}%)\n")

        f.write("\n--- ОТЛАДОЧНЫЙ НАБОР ДАННЫХ (MICRO) ---\n")
        for name, df in micro_splits.items():
            total = df.shape[0]
            attack_count = int(df[target_col].sum()) if target_col in df.columns else 0
            attack_rate = (attack_count / total) * 100 if total > 0 else 0
            display_name = f"{name.capitalize()}_micro"
            f.write(f"-> {display_name:<15}: {total} строк | Атак: {attack_count} шт. ({attack_rate:.2f}%)\n")


    logger.info(f"Паспорт данных успешно сохранен в артефакты: {report_path}")


def run_data_preparation():
    logger.info("=== НАЧАЛО ЭТАПА: ПОДГОТОВКА ДАННЫХ (ДВУХКОНТУРНЫЙ РЕЖИМ) ===")

    raw_dir = BASE_DIR / "data" / "raw"
    processed_dir = BASE_DIR / "data" / "processed"

    path_attack = raw_dir / "text_jailbreak_unique.csv"
    path_normal = raw_dir / "text_regular_unique.csv"

    if not path_attack.exists() or not path_normal.exists():
        logger.error(f"Сырые датасеты не найдены в папке {raw_dir}!")
        sys.exit(1)

    # Загрузка и объединение
    df_attack = pd.read_csv(path_attack, engine="python", on_bad_lines='skip')
    df_normal = pd.read_csv(path_normal, engine="python", on_bad_lines='skip')
    df_combined = pd.concat([df_attack, df_normal])

    # Фильтрация пересечений и дубликатов
    overlap = set(df_attack['user_prompt']).intersection(set(df_normal['user_prompt']))
    if overlap:
        df_combined = df_combined[~df_combined['user_prompt'].isin(overlap)]
    df_combined = df_combined.drop_duplicates(subset=['user_prompt'], keep='first', ignore_index=True)

    # Экстракция мета-признаков
    extractor = TextFeatureExtractor(text_column='user_prompt')
    df_enriched = extractor.transform(df_combined)

    # 1. Генерируем набор сплитов
    logger.info("Генерация основного набора сплитов...")
    stratifier = DataStratifier(target_col='jailbreak', word_len_col='word_len', stress_test_threshold=120)
    full_splits = stratifier.split_data(
        df_enriched)

    # 2. Генерируем отладочный набор micro-сплитов
    logger.info("Генерация отладочного набора micro-сплитов...")
    # Для train_micro используем зафиксированные тобой 1.5% (micro_size=0.015)
    poc_stratifier = DataStratifier(target_col='jailbreak', word_len_col='word_len', stress_test_threshold=120)
    raw_poc_splits = poc_stratifier.split_data(df_enriched, micro_size=0.015)

    # Собираем красивый словарь micro-сплитов
    micro_splits = {
        "train": raw_poc_splits["train_micro"],  # Берем тот самый быстрый train
        "val": full_splits["val"].sample(frac=0.10, random_state=42),  # Бьем валидацию в 10 раз
        "test": full_splits["test"].sample(frac=0.10, random_state=42),  # Бьем тест в 10 раз
        "stress_test": full_splits["stress_test"].sample(frac=0.10, random_state=42)  # Бьем стресс-тест в 10 раз
    }

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем расширенный паспорт данных
    save_split_report(full_splits, micro_splits, processed_dir, target_col='jailbreak')

    # Очистка колонок и сохранение на диск
    cols_to_drop = ['system_prompt', 'source', 'tactic']

    # Сохраняем основной набор сплитов
    for name, df in full_splits.items():
        if name == "train_micro": continue
        save_path = processed_dir / f"{name}.csv"
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore').to_csv(save_path, index=False)

    # Сохраняем набор micro-сплитов
    for name, df in micro_splits.items():
        save_path = processed_dir / f"{name}_micro.csv"
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore').to_csv(save_path, index=False)
        logger.info(f"Сохранен Micro-сплит: {save_path.name} | Размер: {df.shape[0]} строк")

    logger.info("=== ЭТАП ПОДГОТОВКИ ДАННЫХ ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    run_data_preparation()