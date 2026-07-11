# -*- coding: utf-8 -*-
import os
import sys
import shutil
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


def save_split_report(all_splits: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path, target_col: str = 'jailbreak'):
    """
    Создает текстовый паспорт данных, фиксируя объемы для всех экосистем (Full, Micro, Nano).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"data_split_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт выгрузки датасета (Трехконтурный: Full, Micro, Nano)\n")
        f.write(f"Дата и время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Перебираем все режимы: full, micro, nano
        for mode_name, splits_dict in all_splits.items():
            f.write(f"--- НАБОР ДАННЫХ: {mode_name.upper()} ---\n")
            for name, df in splits_dict.items():
                total = df.shape[0]
                attack_count = int(df[target_col].sum()) if target_col in df.columns else 0
                attack_rate = (attack_count / total) * 100 if total > 0 else 0
                f.write(f"-> {name:<12}: {total} строк | Атак: {attack_count} шт. ({attack_rate:.2f}%)\n")
            f.write("\n")

    logger.info(f"Паспорт данных успешно сохранен: {report_path}")


def run_data_preparation():
    logger.info("=== НАЧАЛО ЭТАПА: ПОДГОТОВКА ДАННЫХ (ТРЕХКОНТУРНЫЙ РЕЖИМ) ===")

    raw_dir = BASE_DIR / "data" / "raw"
    processed_dir = BASE_DIR / "data" / "processed"

    # --- БЕЗОПАСНАЯ ОЧИСТКА ПАПКИ PROCESSED ---
    if processed_dir.exists() and processed_dir.is_dir():
        logger.info(f"Очистка предыдущих артефактов в директории: {processed_dir}")
        for item in processed_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Не удалось удалить старый артефакт {item}: {e}")

    # Гарантируем, что папка существует после очистки (или если её не было изначально)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------

    path_attack = raw_dir / "text_jailbreak_unique.csv"
    path_normal = raw_dir / "text_regular_unique.csv"

    if not path_attack.exists() or not path_normal.exists():
        logger.error(f"Сырые датасеты не найдены в папке {raw_dir}!")
        sys.exit(1)

    df_attack = pd.read_csv(path_attack, engine="python", on_bad_lines='skip')
    df_normal = pd.read_csv(path_normal, engine="python", on_bad_lines='skip')
    df_combined = pd.concat([df_attack, df_normal])

    overlap = set(df_attack['user_prompt']).intersection(set(df_normal['user_prompt']))
    if overlap:
        df_combined = df_combined[~df_combined['user_prompt'].isin(overlap)]
    df_combined = df_combined.drop_duplicates(subset=['user_prompt'], keep='first', ignore_index=True)

    extractor = TextFeatureExtractor(text_column='user_prompt')
    df_enriched = extractor.transform(df_combined)

    stratifier = DataStratifier(target_col='jailbreak', word_len_col='word_len', stress_test_threshold=120)

    # 1. Генерируем основной (Full) набор сплитов
    logger.info("Генерация набора: FULL (fraction=1.0)...")
    full_splits = stratifier.split_data(df_enriched, fraction=1.0)

    # 2. Генерируем отладочный (Micro) набор сплитов (~10-15k строк)
    logger.info("Генерация набора: MICRO (fraction=0.015)...")
    micro_splits = stratifier.split_data(df_enriched, fraction=0.015)

    # 3. Генерируем экстремально маленький (Nano) набор для HPO (~350 строк)
    logger.info("Генерация набора: NANO (fraction=0.0005)...")
    nano_splits = stratifier.split_data(df_enriched, fraction=0.0005)

    all_splits_dict = {
        "full": full_splits,
        "micro": micro_splits,
        "nano": nano_splits
    }

    save_split_report(all_splits_dict, processed_dir, target_col='jailbreak')

    cols_to_drop = ['system_prompt', 'source', 'tactic']

    # Динамическое сохранение всех сплитов на диск
    for mode, splits in all_splits_dict.items():
        suffix = "" if mode == "full" else f"_{mode}"

        for name, df in splits.items():
            save_path = processed_dir / f"{name}{suffix}.csv"
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore').to_csv(save_path,
                                                                                                  index=False)
            logger.info(f"Сохранен файл: {save_path.name:<20} | Размер: {df.shape[0]} строк")

    logger.info("=== ЭТАП ПОДГОТОВКИ ДАННЫХ ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    run_data_preparation()