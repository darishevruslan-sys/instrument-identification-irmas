# Экспериментальный протокол

## IRMAS baseline

1. `discover_samples` находит 6705 WAV-фрагментов в 11 каталогах классов.
2. `split_samples(..., validation_size=0.2, random_state=42)` выполняет стратифицированное разбиение 5364/1341.
3. Каждый файл приводится к mono, 22 050 Гц и 3 секундам.
4. Извлекается log Mel-спектрограмма `[1, 128, 130]`, затем z-нормализация одного примера.
5. Medium CNN обучается с `BCEWithLogitsLoss`, class `pos_weight`, augmentation и scheduler.
6. Поклассовые пороги на IRMAS подобраны на той же validation. Полученный F1 — диагностический, не независимый test.

## OpenMIC external evaluation

В OpenMIC используются только 9 точных совпадений классов. Для каждого элемента метрики учитываются только позиции, где `Y_mask=True`; неизвестная метка не превращается в отрицательную.

### IRMAS-only

Порог каждого класса выбирается на официальном OpenMIC train, затем замораживается и применяется к официальному test. Веса IRMAS не дообучаются.

### Fine-tuning

Официальный OpenMIC train делится с seed 42: 80% для fine-tuning и 20% для выбора порогов. Официальный test не участвует ни в обучении, ни в выборе порогов.

## Два checkpoint

- `best_model_mel_v2.pth`: 11-классовый IRMAS baseline, обслуживает веб-демо.
- `best_model_mel_v2_openmic_overlap9.pth`: fine-tuned версия; оценка относится к девяти совпадающим выходам OpenMIC.

Смешивать метрики второго checkpoint с поведением live demo нельзя.
