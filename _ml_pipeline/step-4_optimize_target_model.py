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

project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"



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


def run_optimization():
    # 1. Загрузка конфигурации из .env
    split_group_step4 = os.getenv("STEP4_TARGET_IMPROVING_SPLIT_GROUP", "nano").lower()

    logger.info("=== НАЧАЛО ЭТАПА: УЛУЧШЕНИЕ МОДЕЛИ (HPO + MLFLOW) ===")
    logger.info(f"Целевой контур данных для Шага 4: [{split_group_step4.upper()}]")

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

    # 3. Настройка MLflow (Используем базу данных SQLite локально)
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
    experiment_name = f"Target_Improving_Optuna_{split_group_step4.upper()}"
    mlflow.set_experiment(experiment_name)

    # 4. Чтение Baseline из Шага 3
    base_json_path = target_base_dir / "target_base_metrics.json"
    if not base_json_path.exists():
        logger.error(f"Файл эталона не найден: {base_json_path}! Сначала выполните Шаг 3.")
        sys.exit(1)

    with open(base_json_path, "r", encoding="utf-8") as f:
        step3_data = json.load(f)

    split_group_step3 = step3_data["metadata"]["split_group"]
    step3_hp = step3_data["hyperparameters"]

    logger.info(f"Обнаружен Baseline Шага 3 (Контур: {split_group_step3.upper()})")

    # 5. РАЗВИЛКА АЛГОРИТМА: Оценка Baseline
    if split_group_step3 == split_group_step4:
        logger.info("Сплиты совпадают. Забираем метрики из JSON без повторного обучения.")
        baseline_metrics = step3_data["metrics"]
    else:
        logger.warning(f"Сплиты не совпадают ({split_group_step3} != {split_group_step4}). Пересчитываем Baseline...")
        temp_trainer = ModernBertTrainer(model_name=step3_hp["model_name"], max_length=step3_hp["max_length"])
        temp_trainer.train(
            df_train=df_train,
            epochs=step3_hp["num_train_epochs"],
            batch_size=step3_hp["batch_size"],
            lr=step3_hp.get("learning_rate", 2e-5),
            weight_decay=step3_hp.get("weight_decay", 0.01)
        )
        _, val_dict, _ = temp_trainer.evaluate(df_val, name="Validation")
        _, test_dict, _ = temp_trainer.evaluate(df_test, name="Test")
        _, stress_dict, _ = temp_trainer.evaluate(df_stress, name="Stress-Test")

        baseline_metrics = {
            "val_accuracy": val_dict.get("accuracy", 0.0),
            "val_attack_recall": val_dict.get("1", {}).get("recall", 0.0),
            "test_attack_recall": test_dict.get("1", {}).get("recall", 0.0),
            "stress_test_safe_recall": stress_dict.get("0", {}).get("recall", 0.0)
        }

    # Логируем Baseline в MLflow как отдельный Run для красивого графика
    with mlflow.start_run(run_name="Baseline (Step 3 params)"):
        mlflow.log_params(step3_hp)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})

    # 6. ЗАПУСК OPTUNA HPO
    logger.info("Инициализация автоматического поиска (Optuna)...")
    # Используем токенизатор из временного тренера для консистентности
    tokenizer_for_hpo = ModernBertTrainer(model_name=step3_hp["model_name"]).tokenizer

    hpo_optimizer = ModernBertHPOOptimizer(
        tokenizer=tokenizer_for_hpo,
        model_name=step3_hp["model_name"],
        max_length=step3_hp["max_length"],
        batch_size=step3_hp["batch_size"],
        epochs=step3_hp["num_train_epochs"]
    )

    best_hp = hpo_optimizer.optimize(df_train, df_val, n_trials=5)

    # 7. ФИНАЛЬНОЕ ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ
    logger.info("Старт обучения ФИНАЛЬНОЙ модели на лучших параметрах...")
    final_trainer = ModernBertTrainer(model_name=step3_hp["model_name"], max_length=step3_hp["max_length"])
    final_trainer.train(
        df_train=df_train,
        output_dir=str(target_improving_dir / "checkpoints"),
        epochs=step3_hp["num_train_epochs"],
        batch_size=step3_hp["batch_size"],
        lr=best_hp["learning_rate"],
        weight_decay=best_hp["weight_decay"],
        warmup_ratio=best_hp["warmup_ratio"],
        gradient_accumulation_steps=best_hp["gradient_accumulation_steps"]
    )

    # 8. ОЦЕНКА ФИНАЛЬНОЙ МОДЕЛИ
    val_str, val_dict, _ = final_trainer.evaluate(df_val, name="Validation")
    test_str, test_dict, _ = final_trainer.evaluate(df_test, name="Test")
    stress_str, stress_dict, _ = final_trainer.evaluate(df_stress, name="Stress-Test")

    hpo_metrics = {
        "val_accuracy": val_dict.get("accuracy", 0.0),
        "val_attack_recall": val_dict.get("1", {}).get("recall", 0.0),
        "test_attack_recall": test_dict.get("1", {}).get("recall", 0.0),
        "stress_test_safe_recall": stress_dict.get("0", {}).get("recall", 0.0)
    }

    # 9. СОХРАНЕНИЕ АРТЕФАКТОВ И ОТЧЕТОВ
    final_trainer.save_model(str(target_improving_dir))

    # Сборка JSON
    final_payload = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_group": split_group_step4
        },
        "hyperparameters": {
            **step3_hp,  # базовые неменяющиеся параметры
            **best_hp  # лучшие найденные параметры
        },
        "metrics": hpo_metrics
    }

    json_path = target_improving_dir / "target_improving_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=4, ensure_ascii=False)

    # Формирование текстового отчета со сравнением!
    report_path = target_improving_dir / f"target_improving_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Паспорт качества Target-Improving модели (ModernBERT-base + HPO)\n")
        f.write(f"Дата и время обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Обучающая выборка (Split Group): {split_group_step4.upper()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("=== АНАЛИЗ УЛУЧШЕНИЙ ===\n")
        f.write(f"Baseline Recall атак (Test): {baseline_metrics['test_attack_recall']:.4f}\n")
        f.write(f"HPO Best Recall атак (Test): {hpo_metrics['test_attack_recall']:.4f}\n")

        diff = hpo_metrics['test_attack_recall'] - baseline_metrics['test_attack_recall']
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
    run_optimization()