import chess.engine
from fastapi import FastAPI

from core.config import STOCKFISH_PATH, UNET_CHECKPOINT, YOLO_CHECKPOINT
from services.model_services import ChessModelService
from services.stockfish_service import StockfishService


def on_startup(app: FastAPI) -> None:
    stockfish_service = StockfishService(STOCKFISH_PATH)
    app.state.stockfish_service = stockfish_service

    try:
        app.state.model_service = ChessModelService(
            unet_checkpoint=UNET_CHECKPOINT,
            yolo_checkpoint=YOLO_CHECKPOINT,
        )
    except Exception as exc:
        app.state.model_service = None
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None

    if not STOCKFISH_PATH:
        stockfish_service.error = (
            "STOCKFISH_PATH belum diset. Tambahkan env var STOCKFISH_PATH."
        )
        return

    try:
        stockfish_service.start()
    except FileNotFoundError:
        stockfish_service.error = (
            "Binary Stockfish tidak ditemukan. Cek path pada STOCKFISH_PATH."
        )
    except (OSError, chess.engine.EngineError) as exc:
        stockfish_service.error = f"Gagal inisialisasi Stockfish: {exc}"


def on_shutdown(app: FastAPI) -> None:
    app.state.stockfish_service.close()
