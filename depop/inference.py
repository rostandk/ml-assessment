from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

try:  # pragma: no cover - torch may be unavailable in tests
    import torch
except ModuleNotFoundError:  # type: ignore
    torch = None  # type: ignore
from PIL import Image
try:  # pragma: no cover - optional dependency
    from unsloth import FastVisionModel
except ModuleNotFoundError:  # type: ignore
    FastVisionModel = None  # type: ignore

from .media import MediaCache
from .settings import NotebookSettings


@dataclass
class PredictionRecord:
    product_id: str
    label: int
    image_available: bool
    raw_response: str
    is_spam_pred: bool
    confidence_pred: float
    reason_pred: str


class InferenceRunner:
    """Deterministic Transformers inference with optional micro batching."""

    def __init__(self, settings: NotebookSettings, media_cache: MediaCache):
        self.settings = settings
        self.media_cache = media_cache

    def predict(self, df: pd.DataFrame, micro_batch_size: Optional[int] = None) -> pd.DataFrame:
        if torch is None or FastVisionModel is None:
            raise RuntimeError("Inference dependencies (torch/unsloth) are required")

        model, tokenizer = FastVisionModel.from_pretrained(
            str(self.settings.paths.model_dir),
            load_in_4bit=True,
            dtype=self.settings.training.dtype,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        FastVisionModel.for_inference(model)
        model.eval()

        outputs: list[PredictionRecord] = []
        text_buffer: list[dict[str, Any]] = []
        buffer_records: list[dict[str, Any]] = []

        def flush_buffer() -> None:
            if not text_buffer:
                return
            prompts = [item["prompt"] for item in text_buffer]
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = model.generate(**encoded, max_new_tokens=256, temperature=0.0, do_sample=False)
            decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for rec, decoded in zip(buffer_records, decoded_batch):
                outputs.append(self._build_prediction(rec, decoded, image_available=False))
            text_buffer.clear()
            buffer_records.clear()

        for record in df.to_dict("records"):
            url = record.get("image_url") or ""
            image_path = self.media_cache.download_image(url) if url else None
            has_image = bool(image_path and image_path.exists())
            if has_image or not micro_batch_size:
                flush_buffer()
                prompt, pil_image = self._prepare_prompt(tokenizer, record["description"], image_path)
                encoded = self._encode_prompt(tokenizer, prompt, pil_image, device)
                with torch.no_grad():
                    generated = model.generate(**encoded, max_new_tokens=256, temperature=0.0, do_sample=False)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                outputs.append(self._build_prediction(record, decoded, image_available=has_image))
            else:
                prompt_str = self._prompt_text_only(tokenizer, record["description"])
                text_buffer.append({"prompt": prompt_str, "record": record})
                buffer_records.append(record)
                if len(text_buffer) >= micro_batch_size:
                    flush_buffer()
        flush_buffer()
        return pd.DataFrame([r.__dict__ for r in outputs])

    def _prepare_prompt(
        self,
        tokenizer,
        description: str,
        image_path: Optional[Path],
    ) -> tuple[str, Optional[Image.Image]]:
        user_content = []
        pil_image = None
        note = ""
        if image_path is not None and image_path.exists():
            pil_image = Image.open(image_path).convert("RGB")
            user_content.append({"type": "image"})
        else:
            note = "\n\nNote: image unavailable. Base your judgment on the text only."
        user_content.append(
            {
                "type": "text",
                "text": f"{self.settings.training.prompt_template}\n\nDescription: {description}{note}",
            }
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Strict JSON with is_spam, confidence, reason."}]},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return prompt, pil_image

    def _prompt_text_only(self, tokenizer, description: str) -> str:
        user_content = [
            {
                "type": "text",
                "text": f"{self.settings.training.prompt_template}\n\nDescription: {description}\n\nNote: image unavailable. Base your judgment on the text only.",
            }
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Strict JSON with is_spam, confidence, reason."}]},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _encode_prompt(self, tokenizer, prompt: str, pil_image: Optional[Image.Image], device: torch.device):
        if pil_image is not None:
            encoded = tokenizer(pil_image, prompt, add_special_tokens=False, return_tensors="pt")
            pil_image.close()
        else:
            encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    def _build_prediction(self, record: dict[str, Any], decoded: str, image_available: bool) -> PredictionRecord:
        try:
            payload = json.loads(decoded)
        except Exception:
            payload = {"is_spam": False, "confidence": 0.0, "reason": "malformed"}
        return PredictionRecord(
            product_id=record["product_id"],
            label=int(record.get("label", 0)),
            image_available=image_available,
            raw_response=decoded,
            is_spam_pred=bool(payload.get("is_spam")),
            confidence_pred=float(payload.get("confidence", 0.0)),
            reason_pred=str(payload.get("reason", "")),
        )
