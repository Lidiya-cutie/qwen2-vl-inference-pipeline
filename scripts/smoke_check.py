#!/usr/bin/env python3
"""Offline checks without loading Qwen2-VL weights."""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    for p in [
        ROOT / "src" / "qwen2_inference.py",
        ROOT / "src" / "batch_processor.py",
        ROOT / "scripts" / "run_single.py",
        ROOT / "scripts" / "run_batch.py",
    ]:
        py_compile.compile(str(p), doraise=True)
        print("OK compile", p.name)

    cats = json.loads((ROOT / "configs" / "categories.json").read_text(encoding="utf-8"))
    assert "Лекарственные препараты" in cats

    # Import only helpers without constructing the model
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "qwen2_inference", ROOT / "src" / "qwen2_inference.py"
    )
    # Skip full import if transformers not installed — compile already passed
    try:
        from src.qwen2_inference import Qwen2VLInference

        class _Dummy(Qwen2VLInference):
            def __init__(self):
                pass

        d = _Dummy()
        text = "Категория: Медицинские изделия. Ключевые слова: глюкометр, тест-полоски."
        assert "глюкометр" in d.extract_keywords(text)
        assert "Медицинские изделия" in d.extract_categories(text, cats)
        print("OK extract helpers")
    except ImportError as e:
        print("SKIP model helpers (deps missing):", e)

    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
