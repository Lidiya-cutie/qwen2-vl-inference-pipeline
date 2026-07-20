"""
Batch Image Processor
=====================
Пакетная обработка изображений через Qwen2-VL с сохранением
результатов в CSV и поддержкой докачки (resume).

Author: Lidiya Korotkova
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.qwen2_inference import Qwen2VLInference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORIES = PROJECT_ROOT / "configs" / "categories.json"


class BatchProcessor:
    """Пакетная обработка изображений с resume и логгированием."""

    def __init__(
        self,
        inference: Qwen2VLInference,
        images_folder: str,
        output_file: str,
        prompt: Optional[str] = None,
        categories: Optional[dict] = None,
        categories_path: Optional[str] = None,
        image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        log_file: str = "processing.log",
    ):
        self.inference = inference
        self.images_folder = Path(images_folder)
        self.output_file = output_file
        self.image_extensions = image_extensions
        self.log_file = log_file

        self.prompt = prompt or (
            "Опиши, что изображено на картинке, на русском языке. "
            "Присвой категорию. Определи ключевые слова и фразы для данной категории. "
            "Ответ оформи так: сначала описание, затем строка "
            "'Ключевые слова: ...' и укажи категорию."
        )
        self.categories = categories if categories is not None else self._load_categories(categories_path)
        self.df = self._load_existing()

    def _load_categories(self, categories_path: Optional[str]) -> Dict:
        path = Path(categories_path) if categories_path else DEFAULT_CATEGORIES
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Batch] Категории: {path} ({len(data)})")
            return data
        print(f"[Batch] categories.json не найден ({path}), без словаря категорий")
        return {}

    def _load_existing(self) -> pd.DataFrame:
        if os.path.exists(self.output_file):
            df = pd.read_csv(self.output_file)
            print(f"[Batch] Загружено {len(df)} уже обработанных изображений")
            return df
        return pd.DataFrame(
            columns=["image", "category", "keywords", "output_text", "timestamp"]
        )

    def _processed_files(self) -> set:
        return set(self.df["image"].tolist()) if "image" in self.df.columns else set()

    def _log(self, message: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")

    def run(self, max_new_tokens: int = 400):
        """Запустить пакетную обработку."""
        if not self.images_folder.is_dir():
            raise FileNotFoundError(f"Нет папки изображений: {self.images_folder}")

        processed = self._processed_files()
        image_files = sorted(
            f
            for f in self.images_folder.iterdir()
            if f.suffix.lower() in self.image_extensions
        )
        new_rows = []

        print(f"[Batch] Файлов в папке: {len(image_files)}, уже обработано: {len(processed)}")

        for img_path in image_files:
            if img_path.name in processed:
                continue

            try:
                output = self.inference.generate(img_path, self.prompt, max_new_tokens)
                keywords = self.inference.extract_keywords(output)
                category = self.inference.extract_categories(output, self.categories)

                row = {
                    "image": img_path.name,
                    "category": category,
                    "keywords": keywords,
                    "output_text": output,
                    "timestamp": datetime.now().isoformat(),
                }
                new_rows.append(row)
                print(f"  OK: {img_path.name} -> {category}")
                self._log(f"OK: {img_path.name} | {category}")

            except Exception as e:
                print(f"  FAIL: {img_path.name} -> {e}")
                self._log(f"FAIL: {img_path.name} -> {e}")

        if new_rows:
            out_path = Path(self.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(self.output_file, index=False)
            print(f"\n[Batch] Обработано: {len(new_rows)}. Всего: {len(self.df)}")
        else:
            print("\n[Batch] Новых файлов нет (всё уже в CSV или папка пуста).")
