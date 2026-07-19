"""Convenience launcher for the IRMAS demo UI.

Usage:
    python -m web.run                  # auto-detect checkpoint in outputs/models/
    python -m web.run --checkpoint outputs/models/best_model_mel_v2.pth
    python -m web.run --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from web.backend import app, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IRMAS instrument-identification demo UI.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Путь к .pth checkpoint. По умолчанию ищется в outputs/models/.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Авто-перезагрузка при изменении кода.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        load_model(args.checkpoint)
    uvicorn.run(
        "web.backend:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
