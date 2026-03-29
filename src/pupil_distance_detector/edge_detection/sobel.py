from __future__ import annotations

import cv2
import numpy as np


def detect_edges_sobel(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
