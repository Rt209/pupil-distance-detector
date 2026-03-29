from __future__ import annotations

import cv2
import numpy as np


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (7, 7),
    sigma_x: float = 0.0,
) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, sigma_x)
