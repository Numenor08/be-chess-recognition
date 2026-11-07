import threading
from pathlib import Path

import chess
import chess.engine
from fastapi import HTTPException


class StockfishService:
    def __init__(self, executable_path: str):
        self.path_configured = bool(executable_path)
        self.executable_path = Path(executable_path).expanduser()
        if not self.executable_path.is_absolute():
            self.executable_path = (
                Path(__file__).resolve().parent.parent / self.executable_path
            ).resolve()
        self.engine = None
        self.error: str | None = None
        self.lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.engine is not None

    def start(self) -> None:
        if not self.executable_path.is_file():
            raise FileNotFoundError(str(self.executable_path))

        engine = chess.engine.SimpleEngine.popen_uci(str(self.executable_path))
        try:
            engine.ping()
        except chess.engine.EngineError:
            engine.quit()
            raise

        self.engine = engine
        self.error = None

    def close(self) -> None:
        if self.engine is None:
            return
        try:
            self.engine.quit()
        except chess.engine.EngineError:
            pass
        finally:
            self.engine = None

    def ping(self) -> bool:
        if self.engine is None:
            return False
        try:
            with self.lock:
                self.engine.ping()
        except chess.engine.EngineError as exc:
            self.engine = None
            self.error = f"Stockfish berhenti: {exc}"
            return False
        return True

    def restart(self) -> bool:
        with self.lock:
            self.close()
            try:
                self.start()
            except (FileNotFoundError, OSError, chess.engine.EngineError) as exc:
                self.error = f"Gagal memulai ulang Stockfish: {exc}"
                return False
        return True

    @staticmethod
    def _score_payload(score_obj: chess.engine.PovScore, turn: bool) -> dict:
        score = score_obj.pov(turn)
        return {"cp": score.score(), "mate": score.mate()}

    def analyze(self, fen: str, depth: int = 12, show_variations: int = 3) -> dict:
        if self.engine is None:
            raise HTTPException(
                status_code=503,
                detail=self.error or "Stockfish belum siap",
            )

        if show_variations < 1 or show_variations > 5:
            raise HTTPException(status_code=422, detail="show_variations harus 1..5")

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

        try:
            with self.lock:
                info = self.engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=show_variations,
                )
        except chess.engine.EngineError as exc:
            self.engine = None
            self.error = f"Stockfish berhenti: {exc}"
            if not self.restart():
                raise HTTPException(
                    status_code=503,
                    detail=self.error,
                ) from exc

            try:
                with self.lock:
                    info = self.engine.analyse(
                        board,
                        chess.engine.Limit(depth=depth),
                        multipv=show_variations,
                    )
            except chess.engine.EngineError as retry_exc:
                self.error = f"Stockfish error setelah restart: {retry_exc}"
                raise HTTPException(
                    status_code=503,
                    detail=self.error,
                ) from retry_exc

        info_items = info if isinstance(info, list) else [info]
        top_moves = []
        for idx, item in enumerate(info_items):
            pv_moves = item.get("pv", [])
            top_moves.append(
                {
                    "rank": idx + 1,
                    "move": pv_moves[0].uci() if pv_moves else None,
                    "score": self._score_payload(item["score"], board.turn)
                    if item.get("score")
                    else None,
                    "depth": item.get("depth"),
                    "nodes": item.get("nodes"),
                    "pv": [move.uci() for move in pv_moves],
                }
            )

        primary = info_items[0] if info_items else {}
        primary_pv = primary.get("pv", [])
        return {
            "best_move": primary_pv[0].uci() if primary_pv else None,
            "score": self._score_payload(primary["score"], board.turn)
            if primary.get("score")
            else None,
            "depth": primary.get("depth"),
            "nodes": primary.get("nodes"),
            "pv": [move.uci() for move in primary_pv],
            "top_moves": top_moves,
            "show_variations": show_variations,
        }
