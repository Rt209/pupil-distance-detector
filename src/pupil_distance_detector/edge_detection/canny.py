from __future__ import annotations

import cv2
import numpy as np


def detect_edges_canny(
    image: np.ndarray,
    threshold1: int = 30,
    threshold2: int = 90,
) -> np.ndarray:
    return cv2.Canny(image, threshold1, threshold2)
