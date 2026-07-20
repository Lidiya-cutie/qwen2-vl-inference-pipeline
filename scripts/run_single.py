#!/usr/bin/env python3
"""
Single Image Inference
======================
Обработка одного изображения через Qwen2-VL.

Usage:
    python scripts/run_single.py --image path/to/image.jpg
    python scripts/run_single.py --image path/to/image.jpg --prompt "Describe this image"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qwen2_inference import Qwen2VLInference


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL single image inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--prompt",
        default=(
            "Опиши, что изображено на картинке, на русском языке. "
            "Присвой категорию. Определи ключевые слова и фразы для данной категории."
        ),
        help="Text prompt for the model",
    )
    parser.add_argument("--max-tokens", type=int, default=400, help="Max generation tokens")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HF model id or local path",
    )
    args = parser.parse_args()

    model = Qwen2VLInference(model_name=args.model)
    result = model.generate(args.image, args.prompt, args.max_tokens)

    print("\n=== RESULT ===")
    print(result)
    print("\nKeywords:", model.extract_keywords(result))


if __name__ == "__main__":
    main()
