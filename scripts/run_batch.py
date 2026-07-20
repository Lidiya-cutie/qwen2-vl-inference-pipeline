#!/usr/bin/env python3
"""
Batch Image Processing
======================
Пакетная обработка изображений с resume.

Usage:
    python scripts/run_batch.py --input /path/to/images/ --output results/out.csv
    python scripts/run_batch.py --input /path/to/images/ --output results/out.csv --prompt "Custom prompt"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.batch_processor import BatchProcessor
from src.qwen2_inference import Qwen2VLInference


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL batch image processing")
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", default="results/results.csv", help="Output CSV file")
    parser.add_argument("--prompt", default=None, help="Custom prompt (optional)")
    parser.add_argument(
        "--categories",
        default=None,
        help="Path to categories.json (default: configs/categories.json)",
    )
    parser.add_argument("--max-tokens", type=int, default=400, help="Max generation tokens")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="HF model id or local path",
    )
    args = parser.parse_args()

    model = Qwen2VLInference(model_name=args.model)
    processor = BatchProcessor(
        inference=model,
        images_folder=args.input,
        output_file=args.output,
        prompt=args.prompt,
        categories_path=args.categories,
    )
    processor.run(max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
