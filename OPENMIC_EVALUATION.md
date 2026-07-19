# OpenMIC 2018: внешняя оценка

OpenMIC используется как внешний multi-label набор. Оценка ограничена девятью точными совпадениями названий классов: `cello`, `clarinet`, `flute`, `organ`, `piano`, `saxophone`, `trumpet`, `violin`, `voice`. IRMAS `gac` и `gel` не сопоставляются с общим классом OpenMIC `guitar`.

Неизвестные OpenMIC-аннотации исключаются из метрик через `Y_mask` и не считаются отрицательными.

## Структура данных

```text
data/raw/openmic-2018/
├── audio/
├── partitions/
├── class-map.json
└── openmic-2018.npz
```

Скачать набор: <https://zenodo.org/records/1432913>.

## Проверки

```bash
python -m unittest tests.test_openmic -v
python -m src.evaluate_openmic --max-files 32 --batch-size 8 \
  --sweep-thresholds --optimize-class-thresholds \
  --prefix openmic_overlap9_smoke
```

Smoke-команда с подбором порогов на анализируемом подмножестве служит только диагностикой и не должна цитироваться как test-результат.

## Независимая оценка IRMAS-only

Пороги выбираются на OpenMIC train, затем фиксируются для test:

```bash
python -m src.evaluate_openmic_fixed \
  --batch-size 16 --prefix openmic_overlap9_fixed
```

Результат на `split01_test`: micro-F1 `0,6542`, macro-F1 `0,6443`, precision `0,5461`, recall `0,8156`.

## Fine-tuning

```bash
python -m src.finetune_openmic_overlap \
  --epochs 5 --batch-size 32 --learning-rate 0.0001 \
  --random-state 42 --prefix openmic_overlap9_finetune
```

Fine-tuning выполняется на 80% официального OpenMIC train. Оставшиеся 20% train используются для выбора поклассовых порогов. Официальный test остаётся независимым.

Результат на `split01_test`: micro-F1 `0,6929`, macro-F1 `0,6877`, precision `0,5796`, recall `0,8613`.

Подробные оговорки: [RESULTS.md](RESULTS.md).
