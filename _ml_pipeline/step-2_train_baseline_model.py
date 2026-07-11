# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import joblib
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Гарантируем корректные импорты из src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.train_baseline import BaseLineTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model_report(trainer: BaseLineTrainer, splits: dict, output_dir: Path, split_group: str):
    """
    Генерирует и сохраняет текстовый паспорт качества модели.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"baseline_performance_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Baseline модели (TF-IDF + Logistic Regression)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Обучающая выборка (Split Group): {split_group.upper()}\n")
        f.write("=" * 70 + "\n\n")

        for name, df in splits.items():
            f.write(f"=== Результаты для выборки: {name} ===\n")
            X_eval = trainer._prepare_features(df, is_train=False)
            y_true = df['jailbreak'].values
            preds = trainer.model.predict(X_eval)

            if name == "Stress-Test":
                report = classification_report(y_true, preds, labels=[0], target_names=["Safe (0)"], zero_division=0)
            else:
                report = classification_report(y_true, preds, target_names=["Safe (0)", "Attack (1)"], zero_division=0)

            f.write(report)
            f.write("\n" + "-" * 50 + "\n\n")

    logger.info(f"Текстовый паспорт качества успешно сохранен: {report_path}")


def save_metrics_json(trainer: BaseLineTrainer, val_res: dict, test_res: dict, stress_res: dict, output_dir: Path,
                      split_group: str):
    """
    Экспортирует использованные гиперпараметры и ключевые метрики в JSON-файл для дальнейшего авто-сравнения.
    """
    json_path = output_dir / "baseline_metrics.json"

    # Безопасное извлечение метрик (защита от KeyError)
    val_attack_recall = val_res["report_dict"].get("1", {}).get("recall", 0.0)
    test_attack_recall = test_res["report_dict"].get("1", {}).get("recall", 0.0)
    stress_safe_recall = stress_res["report_dict"].get("0", {}).get("recall", 0.0)
    val_accuracy = val_res["report_dict"].get("accuracy", 0.0)

    payload = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_group": split_group
        },
        "hyperparameters": {
            "model_type": "TF-IDF + Logistic Regression",
            "tfidf_max_features": trainer.vectorizer.max_features,
            "tfidf_ngram_range": list(trainer.vectorizer.ngram_range),
            "logreg_max_iter": trainer.model.max_iter,
            "logreg_C": trainer.model.C,
            "random_state": trainer.model.random_state
        },
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


def run_baseline_training():
    # Чтение конфигурации из .env (по умолчанию nano)
    split_group = os.getenv("STEP2_BASELINE_SPLIT_GROUP", "nano").lower()

    logger.info(f"=== НАЧАЛО ЭТАПА: ОБУЧЕНИЕ BASELINE (TF-IDF + LOGREG) ===")
    logger.info(f"Целевой контур данных: [{split_group.upper()}]")

    processed_dir = BASE_DIR / "data" / "processed"
    baseline_dir = BASE_DIR / "models" / "baseline"

    # --- БЕЗОПАСНАЯ ОЧИСТКА ПАПКИ С АРТЕФАКТАМИ BASELINE ---
    if baseline_dir.exists() and baseline_dir.is_dir():
        logger.info(f"Очистка предыдущих артефактов в директории: {baseline_dir}")
        for item in baseline_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Не удалось удалить старый артефакт {item}: {e}")

    # Создаем папку, если она была удалена или еще не существовала
    baseline_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------

    # Динамическое формирование путей к файлам на основе выбранного контура
    suffix = "" if split_group == "full" else f"_{split_group}"

    train_path = processed_dir / f"train{suffix}.csv"
    val_path = processed_dir / f"val{suffix}.csv"
    test_path = processed_dir / f"test{suffix}.csv"
    stress_path = processed_dir / f"stress_test{suffix}.csv"

    if not train_path.exists():
        logger.error(f"Файл {train_path.name} не найден! Убедитесь, что step-1 был запущен корректно.")
        sys.exit(1)

    # 1. Загрузка данных
    logger.info(f"Загрузка {split_group}-сплитов из папки processed...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_stress = pd.read_csv(stress_path)

    # 2. Инициализация тренера
    trainer = BaseLineTrainer(random_state=42)

    # 3. Обучение модели
    logger.info(f"Запуск обучения модели на Train выборке ({df_train.shape[0]} строк)...")
    trainer.train(df_train, target_col='jailbreak')

    # 4. Вывод метрик в консоль и сбор данных для JSON
    logger.info("Расчет метрик качества...")
    val_res = trainer.evaluate(df_val, name="Validation", target_col='jailbreak')
    test_res = trainer.evaluate(df_test, name="Test", target_col='jailbreak')
    stress_res = trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak')

    # 5. Сохранение метрик на диск (Text + JSON)
    eval_splits = {
        "Validation": df_val,
        "Test": df_test,
        "Stress-Test": df_stress
    }
    save_model_report(trainer, eval_splits, baseline_dir, split_group)
    save_metrics_json(trainer, val_res, test_res, stress_res, baseline_dir, split_group)

    # 6. Сохранение артефактов весов (.pkl)
    logger.info("Сохранение артефактов baseline-модели на диск...")
    vectorizer_path = baseline_dir / "tfidf_vectorizer.pkl"
    model_path = baseline_dir / "logreg_model.pkl"

    joblib.dump(trainer.vectorizer, vectorizer_path)
    joblib.dump(trainer.model, model_path)

    logger.info(f"Векторизатор сохранен в: {vectorizer_path}")
    logger.info(f"Модель сохранена в: {model_path}")
    logger.info("=== ЭТАП ОБУЧЕНИЯ BASELINE ЗАВЕРШЕН УСПЕШНО ===")


if __name__ == "__main__":
    run_baseline_training()