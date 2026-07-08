# -*-coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JailbreakDataset(Dataset):
    """
    Кастомный Dataset для токенизации
    """
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class ModernBertTrainer:
    """
    Пайплайн для fine-tuning и оценки модели ModernBERT
    Для старых моделей GPU необходимо заменить:
        - "bf16=True" на fp16=True"
        - "per_device_train_batch_size=64" на "per_device_train_batch_size=16"
    """
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", num_labels: int = 2):
        logger.info(f"Загрузка токенизатора для {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None

    def train(self, df_train: pd.DataFrame, df_val:pd.DataFrame, output_dir: str = "./results"):
        logger.info("Инициализация модели ModernBERT для классификации...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        # Подготовка датасетов
        train_dataset = JailbreakDataset(
            texts=df_train['user_prompt'].fillna("").tolist(),
            labels=df_train['jailbreak'].tolist(),
            tokenizer=self.tokenizer,
        )
        val_dataset = JailbreakDataset(
            texts=df_val['user_prompt'].fillna("").tolist(),
            labels=df_val['jailbreak'].tolist(),
            tokenizer=self.tokenizer
        )

        # Необходимо переключить настройки в зависимости от GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,   # Для старых моделей GPU
            # per_device_train_batch_size=64,     # Для современных моделей GPU
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,    # Для старых моделей GPU
            # bf16=True,      # Для современных моделей GPU
            logging_steps=100,
            report_to="none"
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        logger.info("Запуск fine-tuning процесса ModernBERT на GPU...")
        trainer.train()
        logger.info("Обучение ModernBERT завершено.")

    def evaluate(self, df_eval: pd.DataFrame, name: str = "Validation"):
        """
        Оценка модели с вычислением вероятностей и метрик
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена!")

        logger.info(f"Запуск инференса ModernBERT на выборке: {name}...")
        eval_dataset = JailbreakDataset(
            texts=df_eval['user_prompt'].fillna("").tolist(),
            labels=df_eval['jailbreak'].tolist() if 'jailbreak' in df_eval.columns else None,
            tokenizer=self.tokenizer
        )

        training_args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=32,
            # fp16=True,    # Для старых моделей GPU
            bf16=True,      # Для современных моделей GPU
            report_to="none"
        )
        trainer = Trainer(model=self.model, args=training_args)

        # Получение логитов
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions

        # Перевод логитов в вероятности через Softmax
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()[:,1]
        preds = np.argmax(logits, axis=-1)

        print(f"\n=== Результаты ModernBERT для выборки: {name} ===")

        if 'jailbreak' in df_eval.columns:
            y_true = df_eval['jailbreak'].values
            try:
                roc_auc = roc_auc_score(y_true, probs)
                roc_auc_str = f"{roc_auc:.4f}"
            except ValueError:
                roc_auc_str = "undefined (only one class present in y_true)"

            report = classification_report(y_true, preds, digits=4, zero_division=0)
            print(f"ROC AUC Score: {roc_auc_str}")
            print("Classification Report:")
            print(report)

        return preds, probs