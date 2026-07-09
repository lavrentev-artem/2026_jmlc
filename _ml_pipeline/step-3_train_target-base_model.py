# -*- coding: utf-8 -*-
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# 1. Автоматический поиск и загрузка переменных из .env в корне проекта
# Сначала ищем .env в корне (на уровень выше папки _ml_pipeline)
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        # Устанавливаем токен в окружение, чтобы библиотеки HF подхватили его автоматически
        os.environ["HF_TOKEN"] = hf_token
        # Отключаем предупреждения о симлинках
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        logging.info("Токен Hugging Face (HF_TOKEN) успешно загружен из .env файла.")
else:
    # Запасной вариант: ищем .env в текущей папке скрипта
    load_dotenv()

# Гарантируем, что корневая папка _ml_pipeline находится в sys.path для импорта из src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.train_modern_bert import ModernBertTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_target_base_report(reports_dict: dict, output_dir: Path, train_split_type: str):
    """
    Создает паспорт качества для ModernBERT с фиксацией времени и типа обучающей выборки.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"target_base_performance_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Target-Base модели (ModernBERT-base)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Обучающая выборка (Split Type): {train_split_type}\n")
        f.write("=" * 70 + "\n\n")

        for name, report_str in reports_dict.items():
            f.write(f"=== Результаты для выборки: {name} ===\n")
            f.write(report_str)
            f.write("\n" + "-" * 50 + "\n\n")

    logger.info(f"Паспорт качества ModernBERT успешно сохранен: {report_path}")


def run_target_base_training(mode: str = "micro", epochs: int = 1, batch_size: int = 8):
    """
    Оркестрирует процесс обучения целевой модели ModernBERT.
    - mode: "full" (обучение на 1.2 млн строк) или "micro" (быстрый прогон на легковесных сплитах).
    - epochs: Количество эпох обучения.
    - batch_size: Размер батча.
    """
    logger.info(f"=== НАЧАЛО ЭТАПА: ОБУЧЕНИЕ TARGET-BASE (MODERNBERT) В РЕЖИМЕ [{mode.upper()}] ===")

    processed_dir = BASE_DIR / "data" / "processed"
    models_dir = BASE_DIR / "models" / "target-base"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Динамически выставляем суффиксы файлов в зависимости от выбранного режима
    suffix = "_micro" if mode == "micro" else ""

    train_path = processed_dir / f"train{suffix}.csv"
    val_path = processed_dir / f"val{suffix}.csv"
    test_path = processed_dir / f"test{suffix}.csv"
    stress_path = processed_dir / f"stress_test{suffix}.csv"

    if not train_path.exists():
        logger.error(f"Файлы выборки для режима {mode} не найдены! Запустите step-1 заново.")
        sys.exit(1)

    # 1. Загрузка данных
    logger.info(f"Загрузка датасетов контура [{mode}] с диска...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_stress = pd.read_csv(stress_path)

    # 2. Инициализация тренера
    trainer = ModernBertTrainer(model_name="answerdotai/ModernBERT-base", max_length=512)

    # 3. Запуск обучения
    logger.info(f"Старт fine-tuning на {df_train.shape[0]} примерах...")
    trainer.train(
        df_train=df_train,
        target_col='jailbreak',
        output_dir=str(models_dir / "checkpoints"),
        epochs=epochs,
        batch_size=batch_size,
        lr=2e-5
    )

    # 4. ПОЛНОЦЕННАЯ ВАЛИДАЦИЯ
    logger.info(f"Расчет финальных метрик качества на отложенных выборках ({mode})...")
    val_report, _ = trainer.evaluate(df_val, name="Validation", target_col='jailbreak', batch_size=32)
    test_report, _ = trainer.evaluate(df_test, name="Test", target_col='jailbreak', batch_size=32)
    stress_report, _ = trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak', batch_size=32)

    # 5. Сохранение текстового паспорта качества
    reports_dict = {
        "Validation": val_report,
        "Test": test_report,
        "Stress-Test": stress_report
    }
    save_target_base_report(reports_dict, models_dir, train_split_type=f"ModernBERT_{mode}")

    # 6. Экспорт финальных весов для воркера
    trainer.save_model(str(models_dir))

    logger.info("=== ЭТАП ОБУЧЕНИЯ TARGET-BASE МОДЕЛИ ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    # mode="micro" - обучение на микро выборке
    run_target_base_training(mode="micro", epochs=1, batch_size=8)