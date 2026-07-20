"""
Qwen2-VL Inference Engine
==========================
Базовый модуль для работы с Qwen2-VL-7B-Instruct:
загрузка модели, обработка изображений, генерация описаний.

Author: Lidiya Korotkova
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VLInference:
    """Инференс-движок для Qwen2-VL-7B-Instruct."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Qwen2VL] Loading model {model_name} on {self.device}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("[Qwen2VL] Model loaded.")

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
    ) -> str:
        """Сгенерировать описание для одного изображения."""
        image_ref: Any
        if isinstance(image, Image.Image):
            image_ref = image
        else:
            image_ref = str(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_ref},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    def extract_keywords(self, text: str) -> str:
        match = re.search(r"Ключевые слова:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")
        return ""

    def extract_categories(self, text: str, categories: dict) -> str:
        """Match by category name and optional keywords from configs/categories.json."""
        matched = []
        for cat, meta in categories.items():
            if re.search(rf"\b{re.escape(cat)}\b", text, re.IGNORECASE):
                matched.append(cat)
                continue
            keywords = meta.get("keywords", []) if isinstance(meta, dict) else []
            if any(re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE) for kw in keywords):
                matched.append(cat)
        return ", ".join(dict.fromkeys(matched))
