import base64
import io

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_debug_image_base64(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    warped: np.ndarray,
    corners: np.ndarray,
    validated_grid: dict,
    fen: str,
    board_size: int = 800,
) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Chess Recognition Pipeline Dashboard\nFEN: {fen}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("1. Original Image", fontsize=10)
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("2. Cleaned Mask (U-Net)", fontsize=10)

    contour_vis = orig_rgb.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(contour_vis, [pts], True, (255, 0, 0), 4)
    for i, corner in enumerate(corners):
        cx, cy = int(corner[0]), int(corner[1])
        cv2.circle(contour_vis, (cx, cy), 12, (0, 255, 0), -1)
        cv2.putText(
            contour_vis, str(i), (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3,
        )
    axes[0, 2].imshow(contour_vis)
    axes[0, 2].set_title(f"3. Detection Points (Found: {len(corners)})", fontsize=10)

    det_vis = orig_rgb.copy()
    for (_, _), det in validated_grid.items():
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = (50, 255, 50) if det.color == "w" else (80, 140, 255)
        if getattr(det, "uncertain", False):
            color = (255, 165, 0)
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), color, 3)
        cv2.circle(det_vis, (int((x1 + x2) / 2), int(y2)), 6, (255, 0, 0), -1)
    axes[1, 0].imshow(det_vis)
    axes[1, 0].set_title("4. YOLO Pieces & Basepoints", fontsize=10)

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    cell = board_size / 8
    for i in range(9):
        cv2.line(warped_rgb, (0, int(i * cell)), (board_size, int(i * cell)), (255, 255, 0), 1)
        cv2.line(warped_rgb, (int(i * cell), 0), (int(i * cell), board_size), (255, 255, 0), 1)
    axes[1, 1].imshow(warped_rgb)
    axes[1, 1].set_title("5. Warped Board (Perspective Corrected)", fontsize=10)

    ax = axes[1, 2]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_title("6. FEN Result", fontsize=10)
    symbols = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    }
    for rank_idx, rank_str in enumerate(fen.split(" ")[0].split("/")):
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
                continue
            ax.add_patch(plt.Rectangle([col, 7 - rank_idx], 1, 1, color="#F0D9B5" if (rank_idx + col) % 2 == 0 else "#B58863"))
            ax.text(col + 0.5, (7 - rank_idx) + 0.5, symbols.get(ch, "?"), ha="center", va="center", fontsize=24, color="white" if ch.isupper() else "black")
            col += 1
    for rank_idx in range(8):
        for col in range(8):
            ax.add_patch(plt.Rectangle([col, 7 - rank_idx], 1, 1, color="#F0D9B5" if (rank_idx + col) % 2 == 0 else "#B58863", zorder=0))
    for i in range(8):
        ax.text(i + 0.5, -0.35, chr(ord("a") + i), ha="center", va="center", fontsize=8, color="gray")
        ax.text(-0.35, i + 0.5, str(i + 1), ha="center", va="center", fontsize=8, color="gray")
    for axis in axes.flat:
        axis.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
