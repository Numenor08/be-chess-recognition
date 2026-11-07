import re

import chess
from fastapi import HTTPException

VALID_VIEW_FROM = {"auto", "white", "black", "left", "right"}
CASTLING_PATTERN = re.compile(r"^(K?Q?k?q?|-)$")
EN_PASSANT_PATTERN = re.compile(r"^(-|[a-h][36])$")


def validate_manual_options(
    active_color: str,
    view_from: str,
    castling_rights: str | None,
    en_passant_square: str | None,
    halfmove_clock: int | None,
    fullmove_number: int | None,
) -> None:
    if active_color not in {"w", "b"}:
        raise HTTPException(status_code=422, detail="active_color harus 'w' atau 'b'")
    if view_from not in VALID_VIEW_FROM:
        raise HTTPException(
            status_code=422,
            detail="view_from harus salah satu dari auto/white/black/left/right",
        )
    if castling_rights is not None and not CASTLING_PATTERN.match(castling_rights):
        raise HTTPException(status_code=422, detail="castling_rights tidak valid")
    if en_passant_square is not None and not EN_PASSANT_PATTERN.match(en_passant_square):
        raise HTTPException(status_code=422, detail="en_passant_square tidak valid")
    if halfmove_clock is not None and halfmove_clock < 0:
        raise HTTPException(status_code=422, detail="halfmove_clock minimal 0")
    if fullmove_number is not None and fullmove_number < 1:
        raise HTTPException(status_code=422, detail="fullmove_number minimal 1")


def ensure_fen_valid(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"FEN tidak valid: {exc}") from exc
    if not board.is_valid():
        raise HTTPException(
            status_code=422,
            detail=(
                "Posisi pada FEN tidak valid untuk dianalisis Stockfish "
                f"(status: {board.status()})"
            ),
        )
    return board
