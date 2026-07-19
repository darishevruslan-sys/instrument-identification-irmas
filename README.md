# Идентификация музыкальных инструментов в полифоническом аудио

Исследовательско-инженерный baseline выпускной квалификационной работы Руслана Даришева. Система преобразует аудио в log Mel-спектрограммы и решает multi-label задачу компактной CNN. В репозитории есть обучение, воспроизводимая оценка, два исследовательских checkpoint, FastAPI-демо и Docker-конфигурация.

> Это завершённый baseline, а не новая архитектура, SOTA-сравнение или доказанный коммерческий продукт. Сценарий автоматического тегирования для музыкального поиска и рекомендаций рассматривается как продуктовая гипотеза.

## Что показано в проекте

- полный путь `audio → resample/mono → окна 3 с → log Mel → CNN → sigmoid → threshold`;
- 11 выходных классов IRMAS и внешняя проверка 9 совпадающих классов на OpenMIC 2018;
- корректная обработка неизвестных OpenMIC-меток через `Y_mask`;
- фиксированный `random_state=42` для разбиений;
- upload-only веб-демо: пользователь загружает свой файл, сервер не раздаёт готовые треки;
- лимит загрузки 40 МБ и анализ длинного файла перекрывающимися окнами;
- CI с компиляцией, unit-тестами, загрузкой checkpoint и синтетическим inference.

## Результаты

| Эксперимент | Разделение и выбор порогов | micro-F1 | macro-F1 |
|---|---|---:|---:|
| IRMAS Mel CNN | validation; пороги подобраны на той же выборке | 0,5896 | 0,5943 |
| IRMAS-only → OpenMIC overlap-9 | пороги: OpenMIC train; отчёт: OpenMIC test | 0,6542 | 0,6443 |
| Fine-tuning → OpenMIC overlap-9 | пороги: holdout из train; отчёт: независимый test | **0,6929** | **0,6877** |

Первую строку следует читать только как диагностическую: это не независимый test. Полный протокол, precision/recall и ограничения приведены в [RESULTS.md](RESULTS.md).

## Данные и лицензии

Датасеты в репозитории не распространяются.

- [IRMAS](https://www.upf.edu/web/mtg/irmas) — набор для распознавания доминирующего инструмента; его условия ограничивают коммерческое использование и распространение.
- [OpenMIC 2018](https://zenodo.org/records/1432913) — внешний multi-label набор; в оценке используются только 9 точных пересечений классов.

Код выпущен под MIT. Веса, обученные с использованием IRMAS, предназначены для исследовательского некоммерческого использования; подробнее в [MODEL_CARD.md](MODEL_CARD.md) и [DATASETS.md](DATASETS.md).

## Быстрый старт

Проверенная среда: Python 3.11. Версии зависимостей зафиксированы.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Для веб-демо:

```bash
python -m pip install -r requirements-deploy.txt
python -m web.run --host 127.0.0.1 --port 8000
```

Затем откройте `http://127.0.0.1:8000`. API:

- `GET /api/model-info`;
- `POST /api/analyze` — первые 3 секунды;
- `POST /api/analyze-full` — весь файл скользящими окнами;
- `POST /api/feature-image` — Mel/MFCC/waveform.

## Docker

```bash
docker build -t irmas-demo .
docker run --rm -p 7860:7860 irmas-demo
```

Проверка: `http://127.0.0.1:7860/api/model-info`.

## Данные для воспроизведения

Ожидаемая локальная структура:

```text
data/raw/
├── IRMAS-TrainingData/
│   ├── cel/ cla/ flu/ gac/ gel/ org/ pia/ sax/ tru/ vio/ voi/
└── openmic-2018/
    ├── audio/
    ├── partitions/
    ├── class-map.json
    └── openmic-2018.npz
```

Обучение основной модели:

```bash
python -m src.train \
  --data-dir data/raw/IRMAS-TrainingData \
  --epochs 50 --feature-type mel --model-size medium \
  --augment --scheduler --random-state 42 \
  --model-path outputs/models/best_model_mel_v2.pth
```

Внешняя оценка и fine-tuning описаны в [OPENMIC_EVALUATION.md](OPENMIC_EVALUATION.md). Новое обучение для конкурсной упаковки не запускалось: опубликованы результаты экспериментов защищённой ВКР.

## Архитектура

`InstrumentCNN(medium)` содержит четыре блока `Conv2d → BatchNorm → ReLU → MaxPool → Dropout`, затем adaptive average pooling и линейный слой. Модель возвращает 11 логитов. Обучение использует `BCEWithLogitsLoss`; sigmoid и пороги применяются на оценке/inference.

Основные параметры: `sr=22050`, `duration=3.0`, `n_fft=2048`, `hop_length=512`, `n_mels=128`, mono, seed 42.

## Структура

```text
src/                 обучение, признаки, модели и оценка
web/                 FastAPI и статический UI
tests/               unit/smoke-проверки
outputs/models/       два небольших исследовательских checkpoint
docs/                 протокол и раскрытие AI-помощи
reports/              компактная машинно-читаемая сводка
.github/workflows/    CI
```

## Авторство и AI-помощь

Постановка задачи, запуск экспериментов, проверка результатов и решения об экспериментальном протоколе принадлежат автору. Codex/ChatGPT использовались для реализации, рефакторинга, отладки, анализа и документации под контролем автора. Полное раскрытие: [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md).

## Контакты

Руслан Даришев — [@lodosmor](https://t.me/lodosmor), `darishev.ruslan@gmail.com`.
