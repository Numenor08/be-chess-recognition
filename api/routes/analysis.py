from fastapi import APIRouter, Request

from schemas.analysis import AnalyzeFenRequest
from utils.validation import ensure_fen_valid

router = APIRouter()


@router.post("/analyze-fen")
def analyze_fen(request: Request, payload: AnalyzeFenRequest):
    ensure_fen_valid(payload.fen)
    analysis = request.app.state.stockfish_service.analyze(
        fen=payload.fen,
        depth=payload.depth,
        show_variations=payload.show_variations,
    )
    return {
        "success": True,
        "fen": payload.fen,
        "stockfish": analysis,
    }
