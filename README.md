# Qwen2-VL Inference Pipeline

**Computer Vision & Multimodal ML Pipeline**

Пайплайн для пакетной обработки изображений с **Qwen2-VL-7B-Instruct** (Alibaba Cloud):
классификация, ключевые слова, описания на русском, resume при пакетном прогоне.

## Возможности

- Инференс одного изображения с кастомным промптом
- Пакетная обработка с resume (докачка в CSV)
- Категории из `configs/categories.json` (regex + keywords)
- Извлечение строки «Ключевые слова: …» из ответа модели
- Экспорт в CSV

## Установка

```bash
cd /opt/my_train_models/github   # или клон репозитория
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Модель подтягивается с Hugging Face при первом запуске (~16 GB VRAM для 7B).

## Использование

```bash
# Одно изображение
python scripts/run_single.py --image sample.jpg

python scripts/run_single.py \
  --image sample.jpg \
  --prompt "Опиши изображение на русском, выдели ключевые элементы"

# Папка изображений
python scripts/run_batch.py \
  --input /path/to/images/ \
  --output results/out.csv

python scripts/run_batch.py \
  --input /path/to/images/ \
  --output results/out.csv \
  --categories configs/categories.json \
  --prompt "Определи категорию: лекарства, аптека, медизделия"
```

## Структура

```
github/
├── src/
│   ├── qwen2_inference.py
│   └── batch_processor.py
├── scripts/
│   ├── run_single.py
│   └── run_batch.py
├── configs/
│   └── categories.json
├── results/
│   └── example_results.csv
├── requirements.txt
└── README.md
```

## Технологии

Qwen2-VL-7B-Instruct · transformers · PyTorch · Pillow · pandas · qwen-vl-utils

## Пример результатов

| Изображение | Категория | Ключевые слова |
| --- | --- | --- |
| 2B180380DE3A2E15.jpg | Лекарственные препараты | аптечная сеть, Omega 3, БАД |
| 2E7014F60E534D71.jpg | Медицинские изделия | глюкометр, тест-полоски |

## Лицензия

MIT
