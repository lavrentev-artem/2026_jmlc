# -*-coding: utf-8 -*-
import logging
from pathlib import Path
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
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {key: val for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = int(self.labels[idx])

        return item


class ModernBertTrainer:
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length

        logger.info(f"Загрузка токенизатора: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info(f"Загрузка весов модели с оптимизацией SDPA: {self.model_name}...")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                attn_implementation="sdpa"
            )
            logger.info("Успешно активирован режим ускоренного внимания SDPA!")
        except Exception as e:
            logger.warning(f"Не удалось включить SDPA ({e}), фоллбек на стандартный режим.")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_fp16 = torch.cuda.is_available() and not self.use_bf16

        precision_str = "bf16 (Ampere+)" if self.use_bf16 else (
            "fp16 (Turing/Pascal)" if self.use_fp16 else "FP32 (CPU/Default)")
        logger.info(f"Аппаратно выбранный режим точности вычислений: {precision_str}")

    def train(self, df_train: pd.DataFrame, target_col: str = 'jailbreak', output_dir: str = "./tmp_checkpoints",
                 epochs: int = 1, batch_size: int = 8, lr: float = 2e-5, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.0, gradient_accumulation_steps: int = 1):
        logger.info(f"Подготовка датасета для обучения ({df_train.shape[0]} строк)...")
        train_dataset = JailbreakDataset(
            texts=df_train['user_prompt'].fillna("").tolist(),
            labels=df_train[target_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=50,
            save_strategy="no",
            eval_strategy="no",
            bf16=self.use_bf16,
            fp16=self.use_fp16,
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        logger.info(
            f"Запуск градиентного спуска (Fine-Tuning). Accumulation: {gradient_accumulation_steps}, Warmup: {warmup_ratio}")
        trainer.train()
        logger.info("Обучение ModernBERT успешно завершено!")

    def evaluate(self, df_eval: pd.DataFrame, name: str = "Validation", target_col: str = 'jailbreak',
                 batch_size: int = 32):
        logger.info(f"Инференс модели на выборке: {name} ({df_eval.shape[0]} строк)...")
        eval_dataset = JailbreakDataset(
            texts=df_eval['user_prompt'].fillna("").tolist(),
            labels=df_eval[target_col].tolist() if target_col in df_eval.columns else None,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        # Динамическое вычисление пути до _ml_pipeline/tmp/tmp_eval
        pipeline_root = Path(__file__).resolve().parent.parent
        tmp_eval_dir = pipeline_root / "tmp" / "tmp_eval"
        tmp_eval_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(tmp_eval_dir),  # Используем вычисленный путь
            per_device_eval_batch_size=batch_size,
            bf16=self.use_bf16,
            fp16=self.use_fp16,
            report_to="none",
            dataloader_num_workers=2
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator
        )

        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions

        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
        preds = np.argmax(logits, axis=-1)

        y_true = df_eval[target_col].values
        try:
            roc_auc = roc_auc_score(y_true, probs)
            roc_auc_str = f"{roc_auc:.4f}"
        except ValueError:
            roc_auc = np.nan
            roc_auc_str = "undefined (only one class present in y_true)"

        logger.info(f"[{name}] ROC AUC Score: {roc_auc_str}")

        if name == "Stress-Test":
            report_str = classification_report(y_true, preds, labels=[0], target_names=["Safe (0)"], zero_division=0)
            # Принудительно генерируем словарь даже для одного класса
            report_dict = classification_report(y_true, preds, labels=[0], target_names=["0"], output_dict=True, zero_division=0)
        else:
            report_str = classification_report(y_true, preds, target_names=["Safe (0)", "Attack (1)"], zero_division=0)
            report_dict = classification_report(y_true, preds, output_dict=True, zero_division=0)

        print(f"\n=== Результаты ModernBERT для выборки: {name} ===\n" + report_str)
        return report_str, report_dict, roc_auc

    def save_model(self, save_dir: str):
        logger.info(f"Сохранение весов и токенизатора ModernBERT в {save_dir}...")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info("Артефакты таргет-модели успешно сохранены!")