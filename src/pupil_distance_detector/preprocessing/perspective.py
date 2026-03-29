from __future__ import annotations

import cv2
import numpy as np


def default_face_quad(width: int, height: int) -> np.ndarray:
    return np.float32(
        [
            [0.1 * width, 0.2 * height],
            [0.9 * width, 0.2 * height],
            [0.92 * width, 0.8 * height],
            [0.08 * width, 0.8 * height],
        ]
    )


def apply_perspective_transform(
    image: np.ndarray,
    src_points: np.ndarray | None = None,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    dst_width, dst_height = output_size or (width, height)

    if src_points is None:
        src_points = default_face_quad(width, height)
    src_points = np.float32(src_points)
    dst_points = np.float32(
        [
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (dst_width, dst_height))
