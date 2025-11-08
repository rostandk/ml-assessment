from __future__ import annotations

import json
from typing import Any, Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:  # pragma: no cover - optional heavyweight deps; be forgiving on import errors
    from datasets import load_dataset as hf_load_dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except Exception:  # type: ignore
    hf_load_dataset = None  # type: ignore
    EarlyStoppingCallback = None  # type: ignore
    SFTTrainer = None  # type: ignore
    SFTConfig = None  # type: ignore
    FastVisionModel = None  # type: ignore
    UnslothVisionDataCollator = None  # type: ignore

try:  # pragma: no cover - torch may not be installed locally
    import torch
except ModuleNotFoundError:  # type: ignore
    torch = None  # type: ignore

from .data import SFTDataset
from .settings import NotebookSettings


class QwenTrainer:
    """Fine-tune Qwen3-VL models via Unsloth + TRL."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def train(self, dataset: SFTDataset) -> Dict[str, Any]:
        if torch is None or None in (hf_load_dataset, SFTTrainer, SFTConfig, FastVisionModel, UnslothVisionDataCollator, EarlyStoppingCallback):
            raise RuntimeError("Training dependencies (torch/unsloth/trl/transformers/datasets) are required")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        batch_size = (
            self.settings.training.batch_size_a100
            if "A100" in gpu_name
            else self.settings.training.batch_size_t4
        )
        grad_accum = (
            self.settings.training.grad_accum_a100
            if "A100" in gpu_name
            else self.settings.training.grad_accum_t4
        )

        train_dataset = hf_load_dataset(
            "json", data_files={"train": str(dataset.train_path), "validation": str(dataset.val_path)}
        )

        model, tokenizer = FastVisionModel.from_pretrained(
            self.settings.training.model_id,
            max_seq_length=self.settings.training.max_seq_len,
            dtype=self.settings.training.dtype,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.settings.training.lora_r,
            lora_alpha=self.settings.training.lora_alpha,
            lora_dropout=self.settings.training.lora_dropout,
        )
        FastVisionModel.for_training(model)

        training_config = SFTConfig(
            output_dir=str(self.settings.paths.trainer_dir),
            num_train_epochs=self.settings.training.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.settings.training.learning_rate,
            warmup_ratio=self.settings.training.warmup_ratio,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            bf16=self.settings.training.dtype == "bfloat16",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=256,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=self.settings.training.max_seq_len,
        )

        vision_collator = UnslothVisionDataCollator(model, tokenizer)

        def compute_metrics(eval_preds):
            preds = eval_preds.predictions
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            def extract_label(text: str) -> int:
                try:
                    payload = json.loads(text)
                    return int(bool(payload.get("is_spam")))
                except Exception:
                    return 0

            pred_labels = [extract_label(t) for t in decoded_preds]
            y_true = dataset.val_labels[: len(pred_labels)]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, pred_labels, average="macro", zero_division=0
            )
            accuracy = accuracy_score(y_true, pred_labels)
            return {
                "accuracy": accuracy,
                "macro_f1": f1,
                "macro_precision": precision,
                "macro_recall": recall,
            }

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_config,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["validation"],
            data_collator=vision_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        train_result = trainer.train()
        FastVisionModel.for_inference(model)
        self.settings.paths.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.settings.paths.model_dir))
        tokenizer.save_pretrained(str(self.settings.paths.model_dir))
        return {"train_result": train_result.metrics, "model_dir": self.settings.paths.model_dir}
