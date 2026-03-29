from __future__ import annotations

import cv2
import numpy as np


def apply_binary_threshold(
    image: np.ndarray,
    threshold: int = 55,
    max_value: int = 255,
    invert: bool = True,
) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(image, threshold, max_value, threshold_type)
    return binary


def apply_adaptive_threshold(
    image: np.ndarray,
    max_value: int = 255,
    block_size: int = 31,
    c: int = 7,
    invert: bool = True,
) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        image,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        c,
    )
