# -*-coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import optuna

# Импортируем датасет из соседнего файла, чтобы не дублировать код
from src.train_modern_bert import JailbreakDataset

logger = logging.getLogger(__name__)


class ModernBertHPOOptimizer:
    def __init__(self, tokenizer, model_name: str, max_length: int = 512, batch_size: int = 8, epochs: int = 1):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_fp16 = torch.cuda.is_available() and not self.use_bf16

    def model_init(self):
        """Функция для инициализации 'чистой' модели перед каждым Trial"""
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2, attn_implementation="sdpa"
            )
        except:
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def compute_metrics(self, eval_pred):
        """Метрика для Optuna (оптимизируем Recall по классу Атак)"""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # Считаем recall для класса 1 (атаки)
        attack_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
        return {"attack_recall": attack_recall}

    def hp_space(self, trial):
        """Пространство поиска гиперпараметров TPE"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.1]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
        }

    def optimize(self, df_train: pd.DataFrame, df_val: pd.DataFrame, n_trials: int = 5) -> dict:
        """Основной цикл автоматического поиска"""
        logger.info(f"Подготовка датасетов для HPO (Train: {df_train.shape[0]}, Val: {df_val.shape[0]})...")

        train_dataset = JailbreakDataset(df_train['user_prompt'].fillna("").tolist(), df_train['jailbreak'].tolist(),
                                         self.tokenizer, self.max_length)
        eval_dataset = JailbreakDataset(df_val['user_prompt'].fillna("").tolist(), df_val['jailbreak'].tolist(),
                                        self.tokenizer, self.max_length)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir="./tmp_hpo",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="epoch",  # Оцениваем модель каждую эпоху для HPO
            save_strategy="no",
            logging_steps=10,
            bf16=self.use_bf16,
            fp16=self.use_fp16,
            report_to="mlflow",  # Автоматическая телеметрия в MLflow
            dataloader_num_workers=2
        )

        trainer = Trainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        logger.info(f"Запуск Optuna Search (trials: {n_trials})...")
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=self.hp_space,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics["eval_attack_recall"]
        )

        logger.info(f"HPO завершен. Лучшие параметры: {best_run.hyperparameters}")
        return best_run.hyperparameters