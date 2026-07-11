# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# 1. Автоматический поиск и загрузка переменных из .env в корне проекта
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        logging.info("Токен Hugging Face (HF_TOKEN) успешно загружен из .env файла.")
else:
    load_dotenv()

# Гарантируем, что корневая папка _ml_pipeline находится в sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.train_modern_bert import ModernBertTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_target_base_report(reports_dict: dict, output_dir: Path, train_split_group: str):
    """
    Создает паспорт качества для ModernBERT с фиксацией времени и типа обучающей выборки.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"target_base_performance_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Target-Base модели (ModernBERT-base)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Обучающая выборка (Split Group): {train_split_group.upper()}\n")
        f.write("=" * 70 + "\n\n")

        for name, report_str in reports_dict.items():
            f.write(f"=== Результаты для выборки: {name} ===\n")
            f.write(report_str)
            f.write("\n" + "-" * 50 + "\n\n")

    logger.info(f"Паспорт качества ModernBERT успешно сохранен: {report_path}")


def save_metrics_json(val_dict: dict, test_dict: dict, stress_dict: dict, hyperparameters: dict, output_dir: Path,
                      split_group: str):
    """
    Экспортирует гиперпараметры и метрики в JSON-файл для автоматического сравнения на Шаге 4.
    """
    json_path = output_dir / "target_base_metrics.json"

    val_attack_recall = val_dict.get("1", {}).get("recall", 0.0)
    test_attack_recall = test_dict.get("1", {}).get("recall", 0.0)

    # Для стресс-теста метка класса "Safe" в словаре будет идти под ключом "0"
    stress_safe_recall = stress_dict.get("0", {}).get("recall", 0.0)
    val_accuracy = val_dict.get("accuracy", 0.0)

    payload = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_group": split_group
        },
        "hyperparameters": hyperparameters,
        "metrics": {
            "val_accuracy": val_accuracy,
            "val_attack_recall": val_attack_recall,
            "test_attack_recall": test_attack_recall,
            "stress_test_safe_recall": stress_safe_recall
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    logger.info(f"JSON-отчет (hyperparameters & metrics) сохранен в: {json_path}")


def run_target_base_training():
    """
    Оркестрирует процесс обучения целевой модели ModernBERT (Target-Base).
    Настройки контура подтягиваются из .env
    """
    # Чтение конфигурации из .env (по умолчанию nano)
    split_group = os.getenv("STEP3_TARGET_BASE_SPLIT_GROUP", "nano").lower()

    logger.info(f"=== НАЧАЛО ЭТАПА: ОБУЧЕНИЕ TARGET-BASE (MODERNBERT) ===")
    logger.info(f"Целевой контур данных: [{split_group.upper()}]")

    processed_dir = BASE_DIR / "data" / "processed"
    target_base_dir = BASE_DIR / "models" / "target-base"

    # --- БЕЗОПАСНАЯ ОЧИСТКА ПАПКИ С АРТЕФАКТАМИ TARGET-BASE ---
    if target_base_dir.exists() and target_base_dir.is_dir():
        logger.info(f"Очистка предыдущих артефактов в директории: {target_base_dir}")
        for item in target_base_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Не удалось удалить старый артефакт {item}: {e}")

    target_base_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------

    # Динамически выставляем суффиксы файлов
    suffix = "" if split_group == "full" else f"_{split_group}"

    train_path = processed_dir / f"train{suffix}.csv"
    val_path = processed_dir / f"val{suffix}.csv"
    test_path = processed_dir / f"test{suffix}.csv"
    stress_path = processed_dir / f"stress_test{suffix}.csv"

    if not train_path.exists():
        logger.error(f"Файлы выборки для контура '{split_group}' не найдены! Запустите step-1 заново.")
        sys.exit(1)

    # 1. Загрузка данных
    logger.info(f"Загрузка датасетов контура [{split_group}] с диска...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_stress = pd.read_csv(stress_path)

    # Задаем гиперпараметры базового запуска
    hyperparams = {
        "model_name": "answerdotai/ModernBERT-base",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 1,  # Для тестов оставляем 1
        "batch_size": 8,
        "max_length": 512
    }

    # 2. Инициализация тренера
    trainer = ModernBertTrainer(model_name=hyperparams["model_name"], max_length=hyperparams["max_length"])

    # 3. Запуск обучения
    logger.info(f"Старт fine-tuning на {df_train.shape[0]} примерах...")
    trainer.train(
        df_train=df_train,
        target_col='jailbreak',
        output_dir=str(target_base_dir / "checkpoints"),
        epochs=hyperparams["num_train_epochs"],
        batch_size=hyperparams["batch_size"],
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"]
    )

    # 4. ПОЛНОЦЕННАЯ ВАЛИДАЦИЯ
    logger.info(f"Расчет финальных метрик качества на отложенных выборках ({split_group})...")
    val_str, val_dict, _ = trainer.evaluate(df_val, name="Validation", target_col='jailbreak', batch_size=32)
    test_str, test_dict, _ = trainer.evaluate(df_test, name="Test", target_col='jailbreak', batch_size=32)
    stress_str, stress_dict, _ = trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak', batch_size=32)

    # 5. Сохранение текстового паспорта качества
    reports_dict = {
        "Validation": val_str,
        "Test": test_str,
        "Stress-Test": stress_str
    }
    save_target_base_report(reports_dict, target_base_dir, train_split_group=split_group)

    # 6. Сохранение метрик и гиперпараметров в JSON
    save_metrics_json(val_dict, test_dict, stress_dict, hyperparams, target_base_dir, split_group)

    # 7. Экспорт финальных весов для воркера
    trainer.save_model(str(target_base_dir))

    logger.info("=== ЭТАП ОБУЧЕНИЯ TARGET-BASE МОДЕЛИ ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    run_target_base_training()