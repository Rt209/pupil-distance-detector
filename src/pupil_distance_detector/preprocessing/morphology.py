from __future__ import annotations

import cv2
import numpy as np


def apply_morphology(
    image: np.ndarray,
    operation: str = "close",
    kernel_size: tuple[int, int] = (5, 5),
    iterations: int = 1,
) -> np.ndarray:
    operation_map = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
    }
    if operation not in operation_map:
        raise ValueError(f"Unsupported morphology operation: {operation}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(image, operation_map[operation], kernel, iterations=iterations)
