# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import mlflow
from dotenv import load_dotenv

# 1. Инициализация глобальных путей в начале файла
project_root = Path(__file__).resolve().parent.parent
tmp_dir = project_root / "_ml_pipeline" / "tmp"
mlflow_db_dir = project_root / "mlflow_db"
env_path = project_root / ".env"


def clear_tmp_dir():
    """
    Очищает всё содержимое _ml_pipeline/tmp/ кроме файла .gitkeep
    (для того чтобы папка оставалась в git).
    """
    if tmp_dir.exists() and tmp_dir.is_dir():
        logger.info(f"Очистка всего содержимого временной папки (кроме .gitkeep):\n    {tmp_dir}")

        for item in tmp_dir.iterdir():
            if item.name == '.gitkeep':
                continue

            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Не удалось удалить объект {item} при очистке tmp: {e}")


# Загрузка переменных окружения
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
else:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.train_modern_bert import ModernBertTrainer
from src.hpo_optimizer import ModernBertHPOOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_extended_metrics(val_dict: dict, test_dict: dict, stress_dict: dict) -> dict:
    """
    Экстрагирует расширенный набор метрик (Recall, Precision, F1) для явного логирования в MLflow.
    """
    return {
        # Валидация (для контроля переобучения в процессе HPO)
        "val_attack_recall": val_dict.get("1", {}).get("recall", 0.0),
        "val_attack_precision": val_dict.get("1", {}).get("precision", 0.0),
        "val_attack_f1": val_dict.get("1", {}).get("f1-score", 0.0),

        # Тест (финальная оценка качества алгоритма)
        "test_attack_recall": test_dict.get("1", {}).get("recall", 0.0),
        "test_attack_precision": test_dict.get("1", {}).get("precision", 0.0),
        "test_attack_f1": test_dict.get("1", {}).get("f1-score", 0.0),

        # Стресс-тест (оценка ложных срабатываний на длинных безопасных текстах)
        "stress_test_safe_recall": stress_dict.get("0", {}).get("recall", 0.0),
        "stress_test_safe_precision": stress_dict.get("0", {}).get("precision", 0.0),
        "stress_test_safe_f1": stress_dict.get("0", {}).get("f1-score", 0.0)
    }


def run_optimization():
    # Загрузка конфигурации размера сплита из .env
    split_group_step4 = os.getenv("STEP4_SPLIT_SIZE", "nano").lower()

    # Загрузка параметра количества итераций для улучшения модели из .env
    try:
        n_trials = int(os.getenv("STEP4_ITERATIONS_COUNT", 5))
    except ValueError:
        logger.warning("Параметр STEP4_ITERATIONS_COUNT имеет некорректный формат. Используем 5 итераций по умолчанию.")
        n_trials = 5

    logger.info("=== НАЧАЛО ЭТАПА: УЛУЧШЕНИЕ МОДЕЛИ (HPO + MLFLOW) ===")
    logger.info(f"Целевой контур данных для Шага 4: [{split_group_step4.upper()}]")
    logger.info(f"Запланированное количество итераций HPO: {n_trials}")

    # Очистка папки tmp перед началом работы
    clear_tmp_dir()

    processed_dir = BASE_DIR / "data" / "processed"
    target_base_dir = BASE_DIR / "models" / "target-base"
    target_improving_dir = BASE_DIR / "models" / "target-improving"

    # --- БЕЗОПАСНАЯ ОЧИСТКА ПАПКИ С АРТЕФАКТАМИ STEP 4 ---
    if target_improving_dir.exists() and target_improving_dir.is_dir():
        logger.info(f"Очистка предыдущих артефактов в директории: {target_improving_dir}")
        for item in target_improving_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Не удалось удалить старый артефакт {item}: {e}")
    target_improving_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------

    # 2. Загрузка данных контура Шага 4
    suffix = "" if split_group_step4 == "full" else f"_{split_group_step4}"
    df_train = pd.read_csv(processed_dir / f"train{suffix}.csv")
    df_val = pd.read_csv(processed_dir / f"val{suffix}.csv")
    df_test = pd.read_csv(processed_dir / f"test{suffix}.csv")
    df_stress = pd.read_csv(processed_dir / f"stress_test{suffix}.csv")

    # 4. Чтение Target-Base модели из Шага 3 и получение СКВОЗНОГО ВРЕМЕНИ ЗАПУСКА
    base_json_path = target_base_dir / "target_base_metrics.json"
    if not base_json_path.exists():
        logger.error(f"Файл эталона не найден: {base_json_path}! Сначала выполните Шаг 3.")
        sys.exit(1)

    with open(base_json_path, "r", encoding="utf-8") as f:
        step3_data = json.load(f)

    split_group_step3 = step3_data["metadata"]["split_group"]
    step3_hp = step3_data["hyperparameters"]

    # ТРЕБОВАНИЕ: Читаем сквозное время запуска. Если его нет — генерируем фоллбек
    run_datetime = step3_data["metadata"].get("run_datetime", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # ТРЕБОВАНИЕ: Формируем имя группы по строгому стандарту
    group_run_name = f"{run_datetime}_Step-4_{split_group_step4.upper()}"
    logger.info(f"Обнаружена Target-Base модель Шага 3. Сквозной ID запуска пайплайна: {run_datetime}")

    # 3. Настройка MLflow (Название эксперимента верхнего уровня)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_dir / 'mlflow.db'}")
    experiment_name = f"Target_Improving_Optuna_{split_group_step4.upper()}"
    mlflow.set_experiment(experiment_name)

    # 5. РАЗВИЛКА АЛГОРИТМА: Оценка Target-Base
    if split_group_step3 == split_group_step4:
        logger.info("Сплиты совпадают. Конвертируем метрики из JSON базового шага...")
        # Безопасно парсим метрики (если в старом JSON не было f1/precision — ставим заглушки)
        target_base_metrics = {
            "val_attack_recall": step3_data["metrics"].get("val_attack_recall", 0.0),
            "val_attack_precision": step3_data["metrics"].get("val_attack_precision", 0.0),
            "val_attack_f1": step3_data["metrics"].get("val_attack_f1", 0.0),
            "test_attack_recall": step3_data["metrics"].get("test_attack_recall", 0.0),
            "test_attack_precision": step3_data["metrics"].get("test_attack_precision", 0.0),
            "test_attack_f1": step3_data["metrics"].get("test_attack_f1", 0.0),
            "stress_test_safe_recall": step3_data["metrics"].get("stress_test_safe_recall", 0.0),
            "stress_test_safe_precision": step3_data["metrics"].get("stress_test_safe_precision", 0.0),
            "stress_test_safe_f1": step3_data["metrics"].get("stress_test_safe_f1", 0.0),
        }
    else:
        logger.warning(
            f"Сплиты не совпадают ({split_group_step3} != {split_group_step4}). Пересчитываем Target-Base...")
        temp_trainer = ModernBertTrainer(model_name=step3_hp["model_name"], max_length=step3_hp["max_length"])

        temp_trainer.train(
            df_train=df_train,
            target_col='jailbreak',
            output_dir=str(target_improving_dir / "tmp_target_base_checkpoints"),
            epochs=step3_hp["num_train_epochs"],
            batch_size=step3_hp["batch_size"],
            lr=step3_hp.get("learning_rate", 2e-5),
            weight_decay=step3_hp.get("weight_decay", 0.01)
        )

        _, val_dict, _ = temp_trainer.evaluate(df_val, name="Validation", target_col='jailbreak')
        _, test_dict, _ = temp_trainer.evaluate(df_test, name="Test", target_col='jailbreak')
        _, stress_dict, _ = temp_trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak')

        target_base_metrics = extract_extended_metrics(val_dict, test_dict, stress_dict)

    # --- СТАРТ РОДИТЕЛЬСКОГО КОНТЕКСТА С ЯВНЫМ ИМЕНЕМ ---
    with mlflow.start_run(run_name=group_run_name):

        # Индекс 1: Логируем Target-Base как вложенный Run
        with mlflow.start_run(run_name="1_Target-Base_Baseline", nested=True):
            mlflow.log_params(step3_hp)
            mlflow.log_metrics(target_base_metrics)

        # Индекс 2: ЗАПУСК OPTUNA HPO (Все триалы автоматически вложатся сюда)
        logger.info("Инициализация автоматического поиска (Optuna)...")
        tokenizer_for_hpo = ModernBertTrainer(model_name=step3_hp["model_name"]).tokenizer

        hpo_optimizer = ModernBertHPOOptimizer(
            tokenizer=tokenizer_for_hpo,
            model_name=step3_hp["model_name"],
            max_length=step3_hp["max_length"],
            batch_size=step3_hp["batch_size"],
            epochs=step3_hp["num_train_epochs"]
        )

        # Пробрасываем динамическое количество итераций из .env
        best_hp = hpo_optimizer.optimize(df_train, df_val, n_trials=n_trials)

        # Индекс 3: ФИНАЛЬНОЕ ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ
        logger.info("Старт обучения ФИНАЛЬНОЙ модели на лучших параметрах...")
        final_trainer = ModernBertTrainer(model_name=step3_hp["model_name"], max_length=step3_hp["max_length"])

        with mlflow.start_run(run_name="2_Final_Optimized_Model", nested=True):
            final_trainer.train(
                df_train=df_train,
                target_col='jailbreak',
                output_dir=str(target_improving_dir / "checkpoints"),
                epochs=step3_hp["num_train_epochs"],
                batch_size=step3_hp["batch_size"],
                lr=best_hp["learning_rate"],
                weight_decay=best_hp["weight_decay"],
                warmup_ratio=best_hp["warmup_ratio"],
                gradient_accumulation_steps=best_hp["gradient_accumulation_steps"]
            )

            # Оценка финальной модели
            val_str, val_dict, _ = final_trainer.evaluate(df_val, name="Validation", target_col='jailbreak')
            test_str, test_dict, _ = final_trainer.evaluate(df_test, name="Test", target_col='jailbreak')
            stress_str, stress_dict, _ = final_trainer.evaluate(df_stress, name="Stress-Test", target_col='jailbreak')

            # ТРЕБОВАНИЕ: Извлекаем расширенные метрики (f1, precision, recall) для MLflow
            hpo_metrics = extract_extended_metrics(val_dict, test_dict, stress_dict)

            mlflow.log_params(best_hp)
            mlflow.log_metrics(hpo_metrics)

    # 9. СОХРАНЕНИЕ АРТЕФАКТОВ И ОТЧЕТОВ
    final_trainer.save_model(str(target_improving_dir))

    # ТРЕБОВАНИЕ: Выгружаем run_datetime в итоговый JSON-паспорт Шага 4
    final_payload = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_datetime": run_datetime,  # Сквозной ID сохранен
            "split_group": split_group_step4
        },
        "hyperparameters": {
            **step3_hp,
            **best_hp
        },
        "metrics": hpo_metrics
    }

    json_path = target_improving_dir / "target_improving_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=4, ensure_ascii=False)

    # Тотальная очистка папки tmp после успешного завершения
    clear_tmp_dir()

    # Формирование текстового отчета
    report_path = target_improving_dir / f"target_improving_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Target-Improving модели (ModernBERT-base + HPO)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Сквозной ID пайплайна (Run DT): {run_datetime}\n")
        f.write(f"Обучающая выборка (Split Group): {split_group_step4.upper()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("=== АНАЛИЗ УЛУЧШЕНИЙ ===\n")
        f.write(f"Target-Base Recall атак (Test): {target_base_metrics['test_attack_recall']:.4f}\n")
        f.write(f"HPO Best Recall атак (Test): {hpo_metrics['test_attack_recall']:.4f}\n")

        diff = hpo_metrics['test_attack_recall'] - target_base_metrics['test_attack_recall']
        if diff > 0:
            f.write(f"ВЕРДИКТ: Успех! Модель улучшена на +{diff:.4f} \n\n")
        else:
            f.write(f"ВЕРДИКТ: Новые параметры не превзошли базовые. Разница: {diff:.4f} \n\n")

        f.write("=== ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ ===\n")
        for k, v in best_hp.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        f.write("=== РЕЗУЛЬТАТЫ ФИНАЛЬНОЙ МОДЕЛИ ===\n")
        f.write("--- Validation ---\n" + val_str + "\n\n")
        f.write("--- Test ---\n" + test_str + "\n\n")
        f.write("--- Stress-Test ---\n" + stress_str + "\n\n")

    logger.info("=== ЭТАП УЛУЧШЕНИЯ МОДЕЛИ ЗАВЕРШЕН ===")


if __name__ == "__main__":
    # Защитная очистка на случай прерванного предыдущего запуска
    clear_tmp_dir()
    run_optimization()