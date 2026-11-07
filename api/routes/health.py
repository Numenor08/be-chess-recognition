import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()

def get_health_status(app):
    model_ready = app.state.model_service is not None
    stockfish_service = app.state.stockfish_service
    stockfish_ready = stockfish_service.ping()

    status = "ok"
    if not model_ready:
        status = "degraded"
    elif not stockfish_ready:
        status = "partial"

    return {
        "status": status,
        "model_ready": model_ready,
        "model_error": app.state.model_error,
        "stockfish_ready": stockfish_ready,
        "stockfish_error": stockfish_service.error,
        "stockfish_path_configured": stockfish_service.path_configured,
    }

@router.get("/health")
def health_check(request: Request):
    return get_health_status(request.app)

@router.get("/health/stream")
async def health_stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            current_status = get_health_status(request.app)
            yield f"data: {json.dumps(current_status)}\n\n"
            await asyncio.sleep(3)
    return StreamingResponse(event_generator(), media_type="text/event-stream")