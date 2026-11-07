import cv2
import numpy as np
import torch
import re
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataclasses import dataclass
from typing import Optional

from services.image_services import ImageServices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.1
UNCERTAIN_THRESHOLD = 0.5
BOARD_SIZE = 800

YOLO_CLASS_MAP = {
    0: ("p", "w"),
    1: ("r", "w"),
    2: ("n", "w"),
    3: ("b", "w"),
    4: ("q", "w"),
    5: ("k", "w"),
    6: ("p", "b"),
    7: ("r", "b"),
    8: ("n", "b"),
    9: ("b", "b"),
    10: ("q", "b"),
    11: ("k", "b"),
}

PIECE_TO_FEN = {
    ("p", "w"): "P",
    ("r", "w"): "R",
    ("n", "w"): "N",
    ("b", "w"): "B",
    ("q", "w"): "Q",
    ("k", "w"): "K",
    ("p", "b"): "p",
    ("r", "b"): "r",
    ("n", "b"): "n",
    ("b", "b"): "b",
    ("q", "b"): "q",
    ("k", "b"): "k",
}

MAX_PIECES = {"p": 8, "r": 3, "n": 3, "b": 3, "q": 2, "k": 1}

VIEW_FROM_ROTATION = {
    "white": 0,
    "black": 2,
    "left": 3,
    "right": 1,
}

AUTO_CONFIDENCE_THRESHOLD = 0.15
BL_TR_AMBIGUOUS_DELTA = 15.0

CASTLING_RIGHTS_PATTERN = re.compile(r"^(K?Q?k?q?|-)$")
EN_PASSANT_PATTERN = re.compile(r"^(-|[a-h][36])$")


@dataclass
class Detection:
    bbox: tuple
    piece_type: str
    color: str
    confidence: float
    uncertain: bool = False
    base_point: Optional[tuple] = None
    grid_col: Optional[int] = None
    grid_row: Optional[int] = None


class ChessModelService:
    def __init__(self, unet_checkpoint: str, yolo_checkpoint: str):
        self.image_service = ImageServices()
        self.unet_model = self.__load_unet(unet_checkpoint)
        self.yolo_model = self.__load_yolo(yolo_checkpoint)

    def __load_unet(self, checkpoint_path: str):
        model = smp.Unet(
            encoder_name="timm-efficientnet-b2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(DEVICE)

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()
        print(f"[UNet] Loaded dari {checkpoint_path}")
        return model

    def __load_yolo(self, checkpoint_path: str):
        model = YOLO(checkpoint_path)
        print(f"[YOLO] Loaded dari {checkpoint_path}")
        return model

    def segment_board(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        transform = A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = self.unet_model(tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask_clean

    def __validate_corners(self, corners: np.ndarray, shape: tuple) -> bool:
        h, w = shape[:2]
        min_dist = min(w, h) * 0.10

        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(corners[i] - corners[j])
                if dist < min_dist:
                    return False

        area = cv2.contourArea(corners.astype(np.int32))
        if area < (w * h * 0.05):
            return False

        return True

    def __order_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        x_sorted = pts[np.argsort(pts[:, 0]), :]

        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]

        rect[0] = left_most[np.argmin(left_most[:, 1]), :]
        rect[3] = left_most[np.argmax(left_most[:, 1]), :]
        rect[1] = right_most[np.argmin(right_most[:, 1]), :]
        rect[2] = right_most[np.argmax(right_most[:, 1]), :]
        return rect

    def find_board_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        for eps_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
            epsilon = eps_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                corners = self.__order_corners(corners)
                if self.__validate_corners(corners, mask.shape):
                    return corners

        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect).astype(np.float32)
        corners = self.__order_corners(corners)
        print("[Corner] Fallback ke minAreaRect")
        return corners

    def detect_pieces(self, image_bgr: np.ndarray) -> list[Detection]:
        results = self.yolo_model(image_bgr, conf=CONF_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in YOLO_CLASS_MAP:
                    continue

                piece_type, color = YOLO_CLASS_MAP[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det = Detection(
                    bbox=(x1, y1, x2, y2),
                    piece_type=piece_type,
                    color=color,
                    confidence=conf,
                    uncertain=(conf < UNCERTAIN_THRESHOLD),
                )
                detections.append(det)
        return detections

    def detect_board_rotation(self, warped_img: np.ndarray, sample: int = 20) -> tuple[int, float]:
        s = sample
        h, w = warped_img.shape[:2]

        corner_brightness = {
            "TL": cv2.cvtColor(warped_img[:s, :s], cv2.COLOR_BGR2GRAY).mean(),
            "TR": cv2.cvtColor(warped_img[:s, w - s :], cv2.COLOR_BGR2GRAY).mean(),
            "BR": cv2.cvtColor(warped_img[h - s :, w - s :], cv2.COLOR_BGR2GRAY).mean(),
            "BL": cv2.cvtColor(warped_img[h - s :, :s], cv2.COLOR_BGR2GRAY).mean(),
        }

        darkest_corner = min(corner_brightness, key=corner_brightness.get)
        rotation_map = {"BL": 0, "TL": 1, "TR": 2, "BR": 3}
        k = rotation_map[darkest_corner]

        vals = list(corner_brightness.values())
        confidence = (max(vals) - min(vals)) / 255.0

        bl_tr_diff = abs(corner_brightness["BL"] - corner_brightness["TR"])
        if bl_tr_diff <= BL_TR_AMBIGUOUS_DELTA:
            print(
                f"[Orientasi] Ambigu BL vs TR (diff={bl_tr_diff:.1f} <= {BL_TR_AMBIGUOUS_DELTA:.1f}). "
                "Fallback k=0, gunakan override manual."
            )
            k = 0
            confidence = min(confidence, 0.05)

        return k, confidence

    def resolve_conflicts(self, detections: list[Detection]) -> dict:
        grid = {}
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for det in sorted_dets:
            key = (det.grid_row, det.grid_col)
            if key not in grid:
                grid[key] = det
            else:
                incumbent = grid[key]
                if incumbent.uncertain and not det.uncertain:
                    grid[key] = det
        return grid

    def validate_chess_rules(self, grid: dict) -> dict:
        counts = {}
        for (row, col), det in grid.items():
            key = (det.piece_type, det.color)
            if key not in counts:
                counts[key] = []
            counts[key].append(((row, col), det))

        invalid_keys = set()
        for (piece_type, color), items in counts.items():
            max_allowed = MAX_PIECES.get(piece_type, 2)
            if len(items) > max_allowed:
                items_sorted = sorted(items, key=lambda x: x[1].confidence)
                excess = len(items) - max_allowed
                for i in range(excess):
                    invalid_keys.add(items_sorted[i][0])

        validated_grid = {k: v for k, v in grid.items() if k not in invalid_keys}

        if len(validated_grid) > 32:
            all_dets = sorted(validated_grid.items(), key=lambda x: x[1].confidence)
            while len(validated_grid) > 32:
                key_to_remove = all_dets.pop(0)[0]
                del validated_grid[key_to_remove]

        return validated_grid

    def _grid_to_algebraic(self, col: int, row: int) -> str:
        file_char = chr(ord("a") + col)
        rank = 8 - row
        return f"{file_char}{rank}"

    def grid_to_fen(
        self,
        validated_grid: dict,
        warped_img: np.ndarray,
        active_color: str = "w",
        view_from: str = "auto",
        castling_rights: Optional[str] = None,
        en_passant_square: Optional[str] = None,
        halfmove_clock: Optional[int] = None,
        fullmove_number: Optional[int] = None,
    ) -> str:
        board = [["." for _ in range(8)] for _ in range(8)]
        for (row, col), det in validated_grid.items():
            fen_char = PIECE_TO_FEN.get((det.piece_type, det.color), ".")
            board[row][col] = fen_char

        if view_from == "auto":
            k, conf = self.detect_board_rotation(warped_img)
            if conf < AUTO_CONFIDENCE_THRESHOLD:
                k = 0
        else:
            k = VIEW_FROM_ROTATION[view_from]

        if k != 0:
            board = np.rot90(np.array(board), k=k).tolist()

        fen_rows = []
        for row in board:
            fen_row = ""
            empty_count = 0
            for cell in row:
                if cell == ".":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        position = "/".join(fen_rows)
        castling, en_passant, halfmove, fullmove = self._normalize_fen_fields(
            castling_rights=castling_rights,
            en_passant_square=en_passant_square,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )

        fen = f"{position} {active_color} {castling} {en_passant} {halfmove} {fullmove}"
        return fen

    def _normalize_fen_fields(
        self,
        castling_rights: Optional[str],
        en_passant_square: Optional[str],
        halfmove_clock: Optional[int],
        fullmove_number: Optional[int],
    ) -> tuple[str, str, int, int]:
        castling = self._normalize_castling_rights(castling_rights)

        en_passant = "-"
        if en_passant_square is not None:
            en_passant = en_passant_square.lower()
            if not EN_PASSANT_PATTERN.fullmatch(en_passant):
                raise ValueError("invalid_en_passant_square")

        halfmove = 0 if halfmove_clock is None else int(halfmove_clock)
        if halfmove < 0:
            raise ValueError("invalid_halfmove_clock")

        fullmove = 1 if fullmove_number is None else int(fullmove_number)
        if fullmove < 1:
            raise ValueError("invalid_fullmove_number")

        return castling, en_passant, halfmove, fullmove

    def _normalize_castling_rights(self, castling_rights: Optional[str]) -> str:
        if castling_rights is None:
            return "-"

        rights = castling_rights.strip()
        if rights == "":
            return "-"

        if rights == "-":
            return rights

        ordered = "".join([c for c in "KQkq" if c in rights])
        if ordered == "":
            raise ValueError("invalid_castling_rights")

        if not CASTLING_RIGHTS_PATTERN.fullmatch(ordered):
            raise ValueError("invalid_castling_rights")

        return ordered

    def predict_from_image(
        self,
        image_bgr: np.ndarray,
        active_color: str = "w",
        view_from: str = "auto",
        castling_rights: Optional[str] = None,
        en_passant_square: Optional[str] = None,
        halfmove_clock: Optional[int] = None,
        fullmove_number: Optional[int] = None,
        debug: bool = False,
    ) -> dict:
        mask = self.segment_board(image_bgr)

        corners = self.find_board_corners(mask)
        if corners is None:
            raise ValueError("corner_detection_failed")

        warped, M, _ = self.image_service.warp_board(image_bgr, corners, size=BOARD_SIZE)
        detections = self.detect_pieces(image_bgr)
        mapped = self.image_service.map_detections_to_grid(detections, M, size=BOARD_SIZE)

        grid = self.resolve_conflicts(mapped)
        validated = self.validate_chess_rules(grid)

        fen = self.grid_to_fen(
            validated,
            warped,
            active_color=active_color,
            view_from=view_from,
            castling_rights=castling_rights,
            en_passant_square=en_passant_square,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )

        result = {
            "fen": fen,
            "num_detected": len(detections),
            "num_validated": len(validated),
            "corners": corners.tolist() if corners is not None else None,
        }

        # Di dalam services/model_services.py pada baris paling akhir sebelum return
        if debug:
            result.update({
                "debug_data": {
                    "image_bgr": image_bgr,
                    "mask": mask,
                    "warped": warped,
                    "corners": corners,
                    "grid": validated
                }
            })
            return result

        return result
