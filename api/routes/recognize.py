import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from utils.validation import validate_manual_options
from utils.visualization import get_debug_image_base64

router = APIRouter()


def _decode_upload_to_image(uploaded_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(uploaded_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="File gambar tidak bisa dibaca")
    return image_bgr


@router.post("/recognize")
async def recognize_position(
    request: Request,
    image: UploadFile = File(...),
    active_color: str = Form("w"),
    view_from: str = Form("auto"),
    castling_rights: str | None = Form(None),
    en_passant_square: str | None = Form(None),
    halfmove_clock: int | None = Form(None),
    fullmove_number: int | None = Form(None),
    debug: bool = Form(False),
):
    app = request.app
    model_service = app.state.model_service
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model belum siap: {app.state.model_error or 'unknown error'}",
        )

    validate_manual_options(
        active_color=active_color,
        view_from=view_from,
        castling_rights=castling_rights,
        en_passant_square=en_passant_square,
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number,
    )

    uploaded_bytes = await image.read()
    if not uploaded_bytes:
        raise HTTPException(status_code=422, detail="File image kosong")

    image_bgr = _decode_upload_to_image(uploaded_bytes)

    try:
        result = model_service.predict_from_image(
            image_bgr=image_bgr,
            active_color=active_color,
            view_from=view_from,
            castling_rights=castling_rights,
            en_passant_square=en_passant_square,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
            debug=debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Pipeline gagal: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    response_payload = {
        "success": True,
        "fen": result["fen"],
        "num_detected": result["num_detected"],
        "num_validated": result["num_validated"],
        "corners": result.get("corners", []),
    }

    if debug and "debug_data" in result:
        dbg = result["debug_data"]
        base64_img = get_debug_image_base64(
            image_bgr=dbg["image_bgr"],
            mask=dbg["mask"],
            warped=dbg["warped"],
            corners=dbg["corners"],
            validated_grid=dbg["grid"],
            fen=result["fen"],
        )
        response_payload["debug_image_base64"] = f"data:image/jpeg;base64,{base64_img}"
    elif "debug" in result:
        response_payload["debug"] = result["debug"]

    return response_payload
