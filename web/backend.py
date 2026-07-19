"""FastAPI backend for the IRMAS instrument-identification demo UI.

Reuses the existing ML pipeline (src.features / src.model / src.utils) so the
predictions shown in the browser come from the real trained model.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.features import (
    compute_mel_spectrogram,
    compute_mfcc,
    load_audio,
    normalize_feature,
)
from src.model import build_model
from src.utils import (
    get_checkpoint_model_size,
    get_device,
    get_state_dict_from_checkpoint,
    load_checkpoint,
)


# ----------------------------------------------------------------------------
# Globals populated in load_model()
# ----------------------------------------------------------------------------
class ModelState:
    """Container for the lazily loaded model + metadata."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.device = get_device()
        self.model_size: str = "small"
        self.feature_type: str = config.DEFAULT_FEATURE_TYPE
        self.checkpoint_path: Path | None = None
        self.checkpoint_message: str = ""
        self.ready: bool = False


STATE = ModelState()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def _default_checkpoint_search() -> Path | None:
    """Search outputs/models for the best available checkpoint."""
    models_dir = config.MODELS_DIR
    if not models_dir.exists():
        return None
    preference = [
        "best_model_mel_v2.pth",
        "best_model_mel_v2_best.pth",
        "best_model_mel.pth",
        "best_model_mel_baseline_v1.pth",
        "best_model.pth",
        "best_model_mfcc.pth",
    ]
    for name in preference:
        candidate = models_dir / name
        if candidate.exists():
            return candidate
    # fall back to any .pth file
    pth_files = sorted(models_dir.glob("*.pth"))
    return pth_files[0] if pth_files else None


def load_model(checkpoint_path: Path | str | None = None) -> None:
    """Load the trained model into the global STATE."""
    if checkpoint_path is None:
        checkpoint_path = _default_checkpoint_search()

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        STATE.ready = False
        STATE.model = None
        STATE.checkpoint_message = (
            "Checkpoint не найден. Положите веса модели в outputs/models/ "
            "(например best_model_mel_v2.pth) или обучите модель: "
            "python -m src.train --data-dir data/raw/IRMAS-TrainingData --epochs 50 "
            "--feature-type mel --model-size medium --augment --scheduler "
            "--model-path outputs/models/best_model_mel_v2.pth"
        )
        return

    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = load_checkpoint(checkpoint_path, STATE.device)
        STATE.model_size = get_checkpoint_model_size(checkpoint)
        model = build_model(num_classes=config.NUM_CLASSES, model_size=STATE.model_size).to(STATE.device)
        model.load_state_dict(get_state_dict_from_checkpoint(checkpoint))
        model.eval()

        # Feature type from checkpoint if present, else default
        if isinstance(checkpoint, dict) and "feature_type" in checkpoint:
            STATE.feature_type = str(checkpoint["feature_type"])
        else:
            STATE.feature_type = config.DEFAULT_FEATURE_TYPE

        STATE.model = model
        STATE.checkpoint_path = checkpoint_path
        STATE.ready = True
        STATE.checkpoint_message = "Model ready"
    except Exception as exc:  # noqa: BLE001
        STATE.ready = False
        STATE.model = None
        STATE.checkpoint_message = f"Ошибка загрузки checkpoint: {exc}"


# ----------------------------------------------------------------------------
# Audio helpers (reuse src.features)
# ----------------------------------------------------------------------------
def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    """Read uploaded bytes and validate the extension."""
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемый формат файла: '{suffix}'. "
                f"Поддерживаются: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )
    # Read at most one byte over the limit. This keeps an oversized upload from
    # being copied into memory in full before the size check runs.
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Получен пустой файл.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой ({len(data) // (1024*1024)} МБ). Максимум {MAX_UPLOAD_BYTES // (1024*1024)} МБ.",
        )
    return data, suffix


def _save_temp_audio(data: bytes, suffix: str) -> Path:
    """Write uploaded bytes to a temp file librosa can read."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def _inspect_raw_audio(file_path: Path) -> dict[str, Any]:
    """Load raw audio without fixed-duration cropping for metadata display."""
    try:
        import librosa

        y, sr = librosa.load(str(file_path), sr=None, mono=False)
        if y.ndim == 1:
            channels = 1
            length = len(y)
        else:
            channels = y.shape[0]
            length = y.shape[1]
        duration = length / sr if sr else 0.0
        return {
            "sample_rate": int(sr),
            "channels": int(channels),
            "length_samples": int(length),
            "duration": round(float(duration), 3),
            "mono": channels == 1,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать аудиофайл: {exc}")


def _run_model(audio: np.ndarray, feature_type: str) -> dict[str, Any]:
    """Extract features from fixed-duration audio and run the CNN."""
    if feature_type == "mel":
        feature = compute_mel_spectrogram(audio)
    elif feature_type == "mfcc":
        feature = compute_mfcc(audio)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    feature_tensor = torch.from_numpy(feature).unsqueeze(0).to(STATE.device)
    with torch.no_grad():
        logits = STATE.model(feature_tensor)
    logits_np = logits.detach().cpu().numpy()[0]
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    return {
        "logits": logits_np.astype(float).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
    }


def _bytes_to_data_url(buf: io.BytesIO, mime: str = "image/png") -> str:
    return "data:" + mime + ";base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _run_model_on_window(audio_window: np.ndarray, feature_type: str) -> np.ndarray:
    """Run the CNN on a single 3-second window and return probabilities (float64)."""
    if feature_type == "mel":
        feature = compute_mel_spectrogram(audio_window)
    elif feature_type == "mfcc":
        feature = compute_mfcc(audio_window)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    feature_tensor = torch.from_numpy(feature).unsqueeze(0).to(STATE.device)
    with torch.no_grad():
        logits = STATE.model(feature_tensor)
    logits_np = logits.detach().cpu().numpy()[0]
    return 1.0 / (1.0 + np.exp(-logits_np))


def _run_sliding_window(
    file_path: Path,
    feature_type: str,
    window_duration: float = config.DURATION,
    hop_duration: float = 1.5,
) -> dict[str, Any]:
    """Slice the entire audio into overlapping windows, run the model on each,
    and return averaged probabilities + per-window details."""
    # Load full audio at the model's target sample rate, mono
    audio_full, sr = librosa.load(str(file_path), sr=config.SAMPLE_RATE, mono=True)
    audio_full = audio_full.astype(np.float32)
    total_duration = len(audio_full) / sr

    window_samples = int(round(sr * window_duration))
    hop_samples = int(round(sr * hop_duration))

    if len(audio_full) < window_samples:
        # file is shorter than one window — just pad and run single window
        padded = np.pad(audio_full, (0, window_samples - len(audio_full)), mode="constant")
        avg_probs = _run_model_on_window(padded, feature_type)
        per_window = [
            {"start": 0.0, "end": window_duration, "probabilities": avg_probs.tolist()}
        ]
    else:
        # collect all windows
        positions: list[int] = []
        pos = 0
        while pos + window_samples <= len(audio_full):
            positions.append(pos)
            pos += hop_samples
        # make sure we also cover the tail
        if positions[-1] + window_samples < len(audio_full):
            positions.append(len(audio_full) - window_samples)

        all_probs = np.zeros((len(positions), config.NUM_CLASSES), dtype=np.float64)
        per_window: list[dict[str, Any]] = []

        for i, start in enumerate(positions):
            window = audio_full[start : start + window_samples]
            probs = _run_model_on_window(window, feature_type)
            all_probs[i] = probs
            per_window.append(
                {
                    "start": round(start / sr, 2),
                    "end": round((start + window_samples) / sr, 2),
                    "probabilities": probs.tolist(),
                }
            )

        avg_probs = all_probs.mean(axis=0)

    # Convert avg_probs → avg_logits (inverse sigmoid) for display
    eps = 1e-7
    avg_probs_clamped = np.clip(avg_probs, eps, 1.0 - eps)
    avg_logits = np.log(avg_probs_clamped / (1.0 - avg_probs_clamped))

    return {
        "probabilities": avg_probs.tolist(),
        "logits": avg_logits.tolist(),
        "total_duration": round(total_duration, 3),
        "window_duration": window_duration,
        "hop_duration": hop_duration,
        "num_windows": len(per_window),
        "per_window": per_window,
    }


def _plot_feature(audio: np.ndarray, feature_type: str) -> str:
    """Render a feature visualization to a base64 data URL."""
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    try:
        if feature_type == "waveform":
            time_axis = np.arange(len(audio)) / config.SAMPLE_RATE
            ax.plot(time_axis, audio, linewidth=0.6, color="#6366f1")
            ax.set_title("Waveform (3.0 с)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Время, с")
            ax.set_ylabel("Амплитуда")
            ax.set_xlim(0, config.DURATION)
        else:
            if feature_type == "mel":
                matrix = librosa.feature.melspectrogram(
                    y=audio,
                    sr=config.SAMPLE_RATE,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                    n_mels=config.N_MELS,
                    power=2.0,
                )
                matrix = librosa.power_to_db(matrix, ref=np.max)
                title = "Mel-spectrogram (log dB, z-нормализованная)"
                y_label = "Mel bins"
            elif feature_type == "mfcc":
                matrix = librosa.feature.mfcc(
                    y=audio,
                    sr=config.SAMPLE_RATE,
                    n_mfcc=config.N_MFCC,
                    n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH,
                )
                title = "MFCC (z-нормализованные)"
                y_label = "MFCC коэффициенты"
            else:
                raise ValueError(f"Unsupported feature_type: {feature_type}")

            im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
            fig.colorbar(im, ax=ax, format="%.1f")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Время (фреймы)")
            ax.set_ylabel(y_label)

        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return _bytes_to_data_url(buf)
    except Exception:
        plt.close(fig)
        raise


# Late import for the plotting helper (kept here for clarity)
import librosa  # noqa: E402


# ----------------------------------------------------------------------------
# Pipeline step builders (detailed, expandable explanations)
# ----------------------------------------------------------------------------
def _mel_or_mfcc_dim(feature_type: str) -> int:
    return config.N_MELS if feature_type == "mel" else config.N_MFCC


def _time_frames() -> int:
    return 1 + int(config.SAMPLE_RATE * config.DURATION) // config.HOP_LENGTH


def _quick_pipeline_steps(
    filename: str,
    raw_meta: dict[str, Any],
    feature_type: str,
    model_size: str,
    classes_sorted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the 8 pipeline steps with short `detail` and long `explanation` blocks."""
    target_samples = int(config.SAMPLE_RATE * config.DURATION)
    frames = _time_frames()
    feat_dim = _mel_or_mfcc_dim(feature_type)
    crop_state = (
        "crop (обрезка)" if raw_meta["duration"] > config.DURATION
        else "pad (дополнение нулями)" if raw_meta["duration"] < config.DURATION
        else "без изменений"
    )
    feat_param = (
        f"n_mels={config.N_MELS} (число мел-полос)" if feature_type == "mel"
        else f"n_mfcc={config.N_MFCC} (число коэффициентов)"
    )

    steps = [
        {
            "step": 1,
            "title": "Аудиофайл загружен",
            "detail": f"{filename} · {raw_meta['duration']} с · {raw_meta['sample_rate']} Гц · {raw_meta['channels']} кан.",
            "explanation": [
                ("Файл «{f}» прочитан библиотекой librosa. Получены исходные параметры:".format(f=filename)),
                ("• Длительность: {dur} с — реальная длина записи.".format(dur=raw_meta['duration'])),
                ("• Sample rate (частота дискретизации): {sr} Гц — сколько отсчётов в секунде хранит файл. "
                 "{sr} Гц означает, что звук зафиксирован с шагом {us:.1f} мкс.".format(
                    sr=raw_meta['sample_rate'], us=1e6 / raw_meta['sample_rate'])),
                ("• Каналы: {ch} ({m}). Стерео ({ch}) на вход модели всё равно будет приведено к моно.".format(
                    ch=raw_meta['channels'], m='моно' if raw_meta['mono'] else 'стерео')),
                ("Всего сырых отсчётов в исходной записи: {n} = длительность × sample rate.".format(
                    n=raw_meta['length_samples'])),
            ],
        },
        {
            "step": 2,
            "title": "Приведение к фиксированной длительности",
            "detail": (
                f"Моно, resample → {config.SAMPLE_RATE} Гц, {crop_state} "
                f"до {config.DURATION} с ({target_samples} сэмплов)"
            ),
            "explanation": [
                ("Модель обучалась на строго 3-секундных фрагментах, поэтому вход нужно привести к "
                 "фиксированному размеру. Это делает src.features.load_audio:"),
                ("1. Стерео → моно: каналы усредняются, чтобы вход был одномерным."),
                ("2. Resample → {sr} Гц: исходная частота {raw} Гц пересчитывается в рабочую {sr} Гц "
                 "(стандарт librosa). Это унифицирует все файлы.".format(
                    sr=config.SAMPLE_RATE, raw=raw_meta['sample_rate'])),
                ("3. {state}: длительность {dur} с приводится ровно к {tgt:.1f} с.".format(
                    state=crop_state, dur=raw_meta['duration'], tgt=config.DURATION)),
                ("   • Если файл длиннее 3 с — берутся первые {tgt:.0f} с, остальное отбрасывается.".format(tgt=config.DURATION)),
                ("   • Если короче — недостающее дополняется нулями (тишиной)."),
                ("Итог — одномерный массив ровно из {n} отсчётов (3 с × {sr} Гц). "
                 "Именно этот массив видит дальнейшая обработка.".format(n=target_samples, sr=config.SAMPLE_RATE)),
            ],
        },
        {
            "step": 3,
            "title": f"Спектральные признаки: {feature_type.upper()}",
            "detail": f"{feature_type.upper()}: n_fft={config.N_FFT}, hop={config.HOP_LENGTH}, {feat_param}, "
                      f"log dB → форма [1, {feat_dim}, {frames}]",
            "explanation": [
                ("Сырой массив отсчётов мало говорит о тембре, поэтому строится спектральное представление "
                 "{ft} через короткое окно (STFT):".format(ft=feature_type.upper())),
                ("• n_fft={n_fft} — размер окна преобразования Фурье ({ms:.1f} мс при {sr} Гц).".format(
                    n_fft=config.N_FFT, ms=1000 * config.N_FFT / config.SAMPLE_RATE, sr=config.SAMPLE_RATE)),
                ("• hop_length={hop} — шаг между окнами ({ms:.1f} мс). Чем меньше, тем больше временных "
                 "фреймов на выходе.".format(hop=config.HOP_LENGTH, ms=1000 * config.HOP_LENGTH / config.SAMPLE_RATE)),
                ("• {param}.".format(param=feat_param)),
                ("Для mel: энергия в каждой полосе переводится в децибелы: "
                 "10·log10(power) (power_to_db), что приближает шкалу к восприятию слуха."),
                ("• Число временных фреймов T = 1 + ⌊длительность·sample_rate / hop⌋ = 1 + ⌊{tgt}·{sr}/{hop}⌋ = {frames}.".format(
                    tgt=int(config.SAMPLE_RATE * config.DURATION), sr=config.SAMPLE_RATE,
                    hop=config.HOP_LENGTH, frames=frames)),
                ("Выходная матрица: [{dim} × {frames}] — {dim} частотных полос, {frames} моментов времени.".format(
                    dim=feat_dim, frames=frames)),
            ],
        },
        {
            "step": 4,
            "title": "Z-нормализация",
            "detail": "Вычитание среднего и деление на std по всему признаку",
            "explanation": [
                ("Спектрограмма нормируется, чтобы вход модели имел нулевое среднее и единичную дисперсию. "
                 "Это ускоряет и стабилизирует обучение (src.features.normalize_feature):"),
                ("Формула: x_norm = (x − μ) / (σ + ε), где μ — среднее по всей матрице, σ — стандартное "
                 "отклонение, ε=1e-8 защищает от деления на ноль."),
                ("После этого большая часть значений лежит примерно в диапазоне [−2, +2], что удобно для "
                 "нейросети (внутренние слои ожидают нормированный вход)."),
                ("На визуализации выше (Mel/MFCC) вы видите именно эту нормированную матрицу — те же данные, "
                 "что попадают в CNN."),
            ],
        },
        {
            "step": 5,
            "title": f"Тензор в CNN (InstrumentCNN, {model_size})",
            "detail": f"InstrumentCNN ({model_size}) · input [1, 1, {feat_dim}, {frames}]",
            "explanation": [
                ("Нормированная матрица превращается в тензор и подаётся в свёрточную сеть "
                 "InstrumentCNN (src.model, размер «{ms}»):".format(ms=model_size)),
                ("• Размер входа: [1, 1, {dim}, {frames}] = [батч=1, канал=1, частота={dim}, время={frames}].".format(
                    dim=feat_dim, frames=frames)),
                ("• small: 3 свёрточных блока (16→32→64 канала), каждый = Conv2d + BatchNorm + ReLU + "
                 "MaxPool + Dropout."),
                ("• medium: 4 блока (16→32→64→128), больше ёмкость — именно эта модель загружена сейчас." if model_size == "medium"
                 else "• {ms}: 3 свёрточных блока (16→32→64 канала).".format(ms=model_size)),
                ("• После свёрток AdaptiveAvgPool сжимает карту до 1×1, затем полносвязный слой (Linear) "
                 "превращает её в 11 чисел (по числу классов инструментов)."),
                ("• Сеть обучалась на multi-label задаче с BCEWithLogitsLoss, поэтому на выходе — сырые "
                 "логиты, а не вероятности."),
            ],
        },
        {
            "step": 6,
            "title": "Logits",
            "detail": "Сырые выходы классификатора: " + ", ".join(
                f"{config.CLASS_NAMES[c['code']]}={c['logit']:+.2f}" for c in classes_sorted[:3]) + ", …",
            "explanation": [
                ("Модель возвращает 11 чисел — по одному на инструмент. Это логиты (сырые выходы "
                 "полносвязного слоя до активации)."),
                ("Логит может быть любым вещественным числом (положительным или отрицательным):"),
                ("• Большой положительный → модель «уверена», что инструмент присутствует."),
                ("• Большой отрицательный → модель «уверена», что инструмент отсутствует."),
                ("• Около нуля → неопределённость."),
                ("Текущие значения (по убыванию):"),
            ] + [
                "  {i}. {name}: logit = {v:+.4f}".format(
                    i=i + 1, name=c['name'], v=c['logit']) for i, c in enumerate(classes_sorted)
            ],
        },
        {
            "step": 7,
            "title": "Sigmoid → probabilities",
            "detail": "Каждый logit ∈ (0,1): top-3 "
                      + ", ".join(f"{c['name']}={c['probability']:.3f}" for c in classes_sorted[:3]),
            "explanation": [
                ("Логиты неудобно интерпретировать, поэтому применяется сигмоида, превращающая каждое "
                 "число в вероятность от 0 до 1:"),
                ("Формула: p = 1 / (1 + e^(−logit)). Например, logit = 2.3 → p ≈ 0.91, logit = −1.0 → p ≈ 0.27."),
                ("Вероятности независимы по классам (multi-label): сумма не обязана равняться 1, потому что "
                 "в полифонической записи инструменты могут звучать одновременно."),
                ("Текущие вероятности (по убыванию):"),
            ] + [
                "  {i}. {name}: p = {v:.4f} ({pct:.1f}%)".format(
                    i=i + 1, name=c['name'], v=c['probability'], pct=c['probability'] * 100)
                for i, c in enumerate(classes_sorted)
            ],
        },
        {
            "step": 8,
            "title": "Threshold → метки",
            "detail": "probability ≥ threshold ⇒ predicted",
            "explanation": [
                ("Последний шаг — превратить вероятности в конкретные предсказания с помощью порога "
                 "(threshold), который задаётся слайдером вверху:"),
                ("Правило: если p(инструмент) ≥ threshold — инструмент считается «предсказанным» "
                 "(predicted), иначе — нет."),
                ("• threshold = 0.5 (по умолчанию) — сбалансированный порог."),
                ("• выше (0.7+) — модель предсказывает только при высокой уверенности, меньше ложных "
                 "срабатываний, но можно что-то пропустить."),
                ("• ниже (0.3) — больше предсказаний, но выше риск ложных."),
                ("Двигайте слайдер — пересчёт меток происходит мгновенно, без повторного запроса к модели "
                 "(вероятности уже посчитаны)."),
            ],
        },
    ]
    return steps


def _full_pipeline_steps(
    filename: str,
    raw_meta: dict[str, Any],
    feature_type: str,
    model_size: str,
    classes_sorted: list[dict[str, Any]],
    sliding_window: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the 8 pipeline steps for sliding-window (full-song) mode."""
    feat_dim = _mel_or_mfcc_dim(feature_type)
    hop = sliding_window["hop_duration"]
    win = sliding_window["window_duration"]
    num_win = sliding_window["num_windows"]
    total = sliding_window["total_duration"]
    feat_param = (
        f"n_mels={config.N_MELS} (число мел-полос)" if feature_type == "mel"
        else f"n_mfcc={config.N_MFCC} (число коэффициентов)"
    )

    return [
        {
            "step": 1,
            "title": "Аудиофайл загружен",
            "detail": f"{filename} · {raw_meta['duration']} с · {raw_meta['sample_rate']} Гц · {raw_meta['channels']} кан.",
            "explanation": [
                "В режиме «вся песня» анализируется не только первые 3 секунды, а вся запись целиком.",
                ("Файл «{f}» прочитан: длительность {dur} с, sample rate {sr} Гц, {ch} кан. "
                 "({m}).".format(f=filename, dur=raw_meta['duration'], sr=raw_meta['sample_rate'],
                                 ch=raw_meta['channels'], m='моно' if raw_meta['mono'] else 'стерео')),
                ("Именно вся эта длительность ({total} с после ресемплинга) будет покрыта окнами.".format(total=total)),
            ],
        },
        {
            "step": 2,
            "title": "Resample и моно",
            "detail": f"Вся аудиодорожка → {config.SAMPLE_RATE} Гц, моно ({total} с)",
            "explanation": [
                "Перед разбивкой на окна вся дорожка приводится к единому формату:",
                ("• Стерео → моно: усреднение каналов."),
                ("• Resample → {sr} Гц — рабочая частота модели. После этого запись занимает {total} с.".format(
                    sr=config.SAMPLE_RATE, total=total)),
                ("Теперь аудио — одномерный массив, который можно нарезать на фрагменты."),
            ],
        },
        {
            "step": 3,
            "title": "Разбивка на окна (sliding window)",
            "detail": f"Окно {win} с, шаг {hop} с → {num_win} окон",
            "explanation": [
                "Поскольку CNN принимает строго 3-секундные фрагменты, вся песня нарезается скользящим окном:",
                ("• Длина окна: {win} с (фиксирована — столько модель видит за раз).".format(win=win)),
                ("• Шаг (hop): {hop} с — на сколько сдвигается окно каждый раз. Окна перекрываются, "
                 "чтобы не потерять события на границах.".format(hop=hop)),
                ("• Перекрытие = окно − шаг = {ov:.1f} с (это {pct:.0f}% окна).".format(
                    ov=win - hop, pct=100 * (win - hop) / win)),
                ("• Для этой записи получилось {n} окон, покрывающих {total} с.".format(n=num_win, total=total)),
                ("Каждое окно обрабатывается независимо — как отдельный 3-секундный фрагмент."),
            ],
        },
        {
            "step": 4,
            "title": f"Признаки каждого окна: {feature_type.upper()}",
            "detail": f"{feature_type.upper()}: n_fft={config.N_FFT}, hop={config.HOP_LENGTH}, "
                      f"{feat_param}, log dB → z-нормализация",
            "explanation": [
                ("Для каждого из {n} окон независимо строится спектрограмма {ft} тем же способом, что и в "
                 "режиме «3 секунды»:".format(n=num_win, ft=feature_type.upper())),
                ("• n_fft={n_fft} (окно STFT, {ms:.1f} мс), hop_length={hop2} (шаг внутри окна).".format(
                    n_fft=config.N_FFT, ms=1000 * config.N_FFT / config.SAMPLE_RATE, hop2=config.HOP_LENGTH)),
                ("• {param}.".format(param=feat_param)),
                ("• log dB (для mel) + z-нормализация: (x−μ)/(σ+ε)."),
                ("Выход: {n} матриц формы [{dim} × T], по одной на окно.".format(n=num_win, dim=feat_dim)),
            ],
        },
        {
            "step": 5,
            "title": f"CNN на каждом окне (InstrumentCNN, {model_size})",
            "detail": f"InstrumentCNN ({model_size}) × {num_win} окон",
            "explanation": [
                ("Каждое окно прогоняется через одну и ту же обученную модель InstrumentCNN ({ms}):".format(ms=model_size)),
                ("• Вход окна: [1, 1, {dim}, T].".format(dim=feat_dim)),
                ("• Сеть выдаёт 11 логитов для каждого окна."),
                ("• Всего прогонов модели: {n} (по числу окон).".format(n=num_win)),
                ("Это самая затратная по времени часть — поэтому длинные песни обрабатываются дольше."),
            ],
        },
        {
            "step": 6,
            "title": "Sigmoid + усреднение по окнам",
            "detail": "Логиты → вероятности по каждому окну → среднее по всем окнам",
            "explanation": [
                "Сначала логиты каждого окна превращаются в вероятности сигмоидой:",
                ("Формула: p = 1 / (1 + e^(−logit)) — независимо для каждого класса и каждого окна."),
                ("Затем по всем {n} окнам берётся среднее по каждому инструменту:".format(n=num_win)),
                ("p̄(инструмент) = (1/N) · Σ p(инструмент, окно i), где N = {n}.".format(n=num_win)),
                ("Такой агрегат показывает, насколько инструмент «в среднем» присутствует во всей записи, "
                 "а не в одном фрагменте."),
            ],
        },
        {
            "step": 7,
            "title": "Средние вероятности",
            "detail": "top-3: " + ", ".join(
                f"{c['name']}={c['probability']:.3f}" for c in classes_sorted[:3]),
            "explanation": [
                ("После усреднения получаем 11 финальных вероятностей (по всем окнам). "
                 "Они отражают присутствие каждого инструмента в записи целиком:"),
            ] + [
                "  {i}. {name}: p̄ = {v:.4f} ({pct:.1f}%)".format(
                    i=i + 1, name=c['name'], v=c['probability'], pct=c['probability'] * 100)
                for i, c in enumerate(classes_sorted)
            ],
        },
        {
            "step": 8,
            "title": "Threshold → метки",
            "detail": "probability ≥ threshold ⇒ predicted (по усреднённым значениям)",
            "explanation": [
                "Финальный шаг аналогичен режиму «3 секунды» — порог применяется к усреднённым вероятностям:",
                ("Если p̄(инструмент) ≥ threshold — инструмент считается присутствующим в записи."),
                ("threshold задаётся слайдером. Двигайте его — пересчёт мгновенный, без повторной обработки "
                 "всей песни."),
                ("Заметьте: усреднение по окнам делает предсказание устойчивее к коротким эпизодам, чем "
                 "анализ одного фрагмента."),
            ],
        },
    ]


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model()
    yield


app = FastAPI(title="IRMAS Instrument Identification Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root() -> Any:
    """Serve the single-page UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html не найден в web/static/.")
    from starlette.responses import HTMLResponse

    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/model-info")
async def model_info() -> dict[str, Any]:
    """Return model status + class list for the UI."""
    return {
        "ready": STATE.ready,
        "message": STATE.checkpoint_message,
        "model_size": STATE.model_size if STATE.ready else None,
        "feature_type": STATE.feature_type if STATE.ready else config.DEFAULT_FEATURE_TYPE,
        "checkpoint": STATE.checkpoint_path.name if STATE.checkpoint_path else None,
        "class_codes": list(config.CLASS_CODES),
        "class_names": {code: config.CLASS_NAMES[code] for code in config.CLASS_CODES},
        "sample_rate": config.SAMPLE_RATE,
        "duration": config.DURATION,
        "n_mels": config.N_MELS,
        "n_mfcc": config.N_MFCC,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "inference_mode": "upload_only",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Full pipeline: metadata + features + logits/probabilities + pipeline steps."""
    if not STATE.ready:
        raise HTTPException(
            status_code=503,
            detail=STATE.checkpoint_message or "Модель не загружена.",
        )

    data, suffix = _read_upload(file)
    tmp_path = _save_temp_audio(data, suffix)

    try:
        raw_meta = _inspect_raw_audio(tmp_path)
        # Fixed-duration audio (crop/pad to 3.0 s, mono, resampled) -- this is what the model sees
        audio = load_audio(tmp_path, sample_rate=config.SAMPLE_RATE, duration=config.DURATION, mono=config.MONO)

        result = _run_model(audio, STATE.feature_type)
        logits = result["logits"]
        probabilities = result["probabilities"]

        # Build per-class rows
        classes = []
        for index, code in enumerate(config.CLASS_CODES):
            classes.append(
                {
                    "code": code,
                    "name": config.CLASS_NAMES[code],
                    "logit": round(float(logits[index]), 5),
                    "probability": round(float(probabilities[index]), 5),
                }
            )
        classes_sorted = sorted(classes, key=lambda c: c["probability"], reverse=True)

        return JSONResponse(
            {
                "filename": file.filename,
                "raw": raw_meta,
                "model": {
                    "sample_rate": config.SAMPLE_RATE,
                    "duration": config.DURATION,
                    "target_samples": int(config.SAMPLE_RATE * config.DURATION),
                    "feature_type": STATE.feature_type,
                    "model_size": STATE.model_size,
                    "input_shape": [1, config.N_MELS if STATE.feature_type == "mel" else config.N_MFCC,
                                    int(1 + int(config.SAMPLE_RATE * config.DURATION) // config.HOP_LENGTH)],
                    "was_cropped": raw_meta["duration"] > config.DURATION,
                    "was_padded": raw_meta["duration"] < config.DURATION,
                },
                "classes": classes_sorted,
                "logits": logits,
                "probabilities": probabilities,
                "pipeline_steps": _quick_pipeline_steps(
                    filename=file.filename,
                    raw_meta=raw_meta,
                    feature_type=STATE.feature_type,
                    model_size=STATE.model_size,
                    classes_sorted=classes_sorted,
                ),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {exc}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/feature-image")
async def feature_image(
    file: UploadFile = File(...),
    feature_type: str = Form("mel"),
) -> JSONResponse:
    """Render a feature visualization (mel / mfcc / waveform) to a PNG data URL."""
    if feature_type not in {"mel", "mfcc", "waveform"}:
        raise HTTPException(status_code=400, detail="feature_type должен быть mel, mfcc или waveform.")
    data, suffix = _read_upload(file)
    tmp_path = _save_temp_audio(data, suffix)
    try:
        audio = load_audio(tmp_path, sample_rate=config.SAMPLE_RATE, duration=config.DURATION, mono=config.MONO)
        data_url = _plot_feature(audio, feature_type)
        return JSONResponse({"image": data_url, "feature_type": feature_type})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ошибка построения визуализации: {exc}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/analyze-full")
async def analyze_full(
    file: UploadFile = File(...),
    hop_duration: float = Form(default=1.5),
) -> JSONResponse:
    """Sliding-window analysis over the entire audio file.

    Slices the audio into overlapping 3-second windows (configurable hop),
    runs the model on each window, and returns averaged probabilities.
    """
    if not STATE.ready:
        raise HTTPException(
            status_code=503,
            detail=STATE.checkpoint_message or "Модель не загружена.",
        )

    if not (0.1 <= hop_duration <= config.DURATION):
        raise HTTPException(
            status_code=400,
            detail=f"hop_duration должен быть от 0.1 до {config.DURATION}.",
        )

    data, suffix = _read_upload(file)
    tmp_path = _save_temp_audio(data, suffix)

    try:
        raw_meta = _inspect_raw_audio(tmp_path)

        # Run sliding window on the full audio
        sw_result = _run_sliding_window(
            tmp_path,
            feature_type=STATE.feature_type,
            window_duration=config.DURATION,
            hop_duration=hop_duration,
        )

        probabilities = sw_result["probabilities"]
        logits = sw_result["logits"]

        classes = []
        for index, code in enumerate(config.CLASS_CODES):
            classes.append(
                {
                    "code": code,
                    "name": config.CLASS_NAMES[code],
                    "logit": round(float(logits[index]), 5),
                    "probability": round(float(probabilities[index]), 5),
                }
            )
        classes_sorted = sorted(classes, key=lambda c: c["probability"], reverse=True)

        return JSONResponse(
            {
                "filename": file.filename,
                "raw": raw_meta,
                "mode": "full",
                "model": {
                    "sample_rate": config.SAMPLE_RATE,
                    "duration": config.DURATION,
                    "target_samples": int(config.SAMPLE_RATE * config.DURATION),
                    "feature_type": STATE.feature_type,
                    "model_size": STATE.model_size,
                    "input_shape": [1, config.N_MELS if STATE.feature_type == "mel" else config.N_MFCC,
                                    int(1 + int(config.SAMPLE_RATE * config.DURATION) // config.HOP_LENGTH)],
                },
                "sliding_window": {
                    "total_duration": sw_result["total_duration"],
                    "window_duration": sw_result["window_duration"],
                    "hop_duration": hop_duration,
                    "num_windows": sw_result["num_windows"],
                },
                "classes": classes_sorted,
                "logits": logits,
                "probabilities": probabilities,
                "pipeline_steps": _full_pipeline_steps(
                    filename=file.filename,
                    raw_meta=raw_meta,
                    feature_type=STATE.feature_type,
                    model_size=STATE.model_size,
                    classes_sorted=classes_sorted,
                    sliding_window={
                        "total_duration": sw_result["total_duration"],
                        "window_duration": sw_result["window_duration"],
                        "hop_duration": hop_duration,
                        "num_windows": sw_result["num_windows"],
                    },
                ),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {exc}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IRMAS instrument-identification demo UI.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Путь к .pth checkpoint. По умолчанию ищется в outputs/models/.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.checkpoint is not None:
        load_model(args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)
