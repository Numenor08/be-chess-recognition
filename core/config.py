import os

from dotenv import load_dotenv

load_dotenv()

UNET_CHECKPOINT = os.getenv("UNET_CHECKPOINT", "")
YOLO_CHECKPOINT = os.getenv("YOLO_CHECKPOINT", "")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
