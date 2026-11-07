import cv2
import numpy as np
from typing import Any

BOARD_SIZE = 800


class ImageServices:
    @staticmethod
    def warp_board(image_bgr: np.ndarray, corners: np.ndarray, size: int = BOARD_SIZE):
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(corners, dst)
        M_inv = cv2.getPerspectiveTransform(dst, corners)
        warped = cv2.warpPerspective(image_bgr, M, (size, size))
        return warped, M, M_inv

    @staticmethod
    def transform_point(pt: tuple, M: np.ndarray) -> tuple:
        pt_arr = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_arr, M)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    @staticmethod
    def map_detections_to_grid(
        detections: list[Any], M: np.ndarray, size: int = BOARD_SIZE,
    ):
        cell_size = size / 8.0
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            base_x = (x1 + x2) / 2.0
            base_y = y1 * 0.2 + y2 * 0.8

            wx, wy = ImageServices.transform_point((base_x, base_y), M)
            det.base_point = (wx, wy)

            wx = max(0, min(size - 1, wx))
            wy = max(0, min(size - 1, wy))

            col = int(wx / cell_size)
            row = int(wy / cell_size)

            det.grid_col = max(0, min(7, col))
            det.grid_row = max(0, min(7, row))

        return detections