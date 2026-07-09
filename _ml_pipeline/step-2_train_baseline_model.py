# -*- coding: utf-8 -*-
import os
import sys
import logging
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report

# Гарантируем корректные импорты из src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.train_baseline import BaseLineTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model_report(trainer: BaseLineTrainer, splits: dict, output_dir: Path):
    """
    Генерирует и сохраняет текстовый паспорт качества модели с метриками
    для всех отложенных выборок.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"baseline_performance_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Baseline модели (TF-IDF + Logistic Regression)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for name, df in splits.items():
            f.write(f"=== Результаты для выборки: {name} ===\n")

            # Подготавливаем фичи через встроенный метод твоего тренера
            X_eval = trainer._prepare_features(df, is_train=False)
            y_true = df['jailbreak'].values

            # Получаем предсказания
            preds = trainer.model.predict(X_eval)

            # Генерируем классический отчет sklearn
            # Учитываем, что на Stress-Test присутствует только класс 0
            if name == "Stress-Test":
                report = classification_report(y_true, preds, labels=[0], target_names=["Safe (0)"], zero_division=0)
            else:
                report = classification_report(y_true, preds, target_names=["Safe (0)", "Attack (1)"], zero_division=0)

            f.write(report)
            f.write("\n" + "-" * 50 + "\n\n")

    logger.info(f"Паспорт качества модели успешно сохранен: {report_path}")


def run_baseline_training():
    logger.info("=== НАЧАЛО ЭТАПА: ОБУЧЕНИЕ BASELINE (TF-IDF + LOGREG) ===")

    processed_dir = BASE_DIR / "data" / "processed"
    models_dir = BASE_DIR / "models" / "baseline"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Проверка наличия подготовленных сплитов данных
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    stress_path = processed_dir / "stress_test.csv"

    if not train_path.exists():
        logger.error(f"Обработанные данные не найдены! Сначала запустите step-1.")
        sys.exit(1)

    # 1. Загрузка данных с диска
    logger.info("Загрузка сплитов данных из папки processed...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_stress = pd.read_csv(stress_path)

    # 2. Инициализация тренера
    trainer = BaseLineTrainer(random_state=42)

    # 3. Обучение модели
    logger.info(f"Запуск обучения модели на Train выборке ({df_train.shape[0]} строк)...")
    trainer.train(df_train, target_col='jailbreak')

    # 4. Вывод метрик в консоль
    logger.info("Расчет метрик качества...")
    trainer.evaluate(df_val, name="Validation", target_col='jailbreak')
    trainer.evaluate(df_test, name="Test", target_col='jailbreak')
    trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak')

    # 5. Сохранение метрик на диск
    eval_splits = {
        "Validation": df_val,
        "Test": df_test,
        "Stress-Test": df_stress
    }
    save_model_report(trainer, eval_splits, models_dir)

    # 6. Сохранение артефактов весов (.pkl)
    logger.info("Сохранение артефактов baseline-модели на диск...")
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    model_path = models_dir / "logreg_model.pkl"

    joblib.dump(trainer.vectorizer, vectorizer_path)
    joblib.dump(trainer.model, model_path)

    logger.info(f"Векторизатор сохранен в: {vectorizer_path}")
    logger.info(f"Модель сохранена в: {model_path}")
    logger.info("=== ЭТАП ОБУЧЕНИЯ BASELINE ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    run_baseline_training()