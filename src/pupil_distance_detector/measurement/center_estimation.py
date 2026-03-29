from __future__ import annotations

import cv2
import numpy as np


def center_from_contour(contour: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def center_from_circle(circle: tuple[int, int, int] | tuple[float, float, float]) -> tuple[int, int]:
    return int(round(circle[0])), int(round(circle[1]))
