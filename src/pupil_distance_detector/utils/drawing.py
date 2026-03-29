from __future__ import annotations

import cv2
import numpy as np

from .image_io import ensure_bgr
from .models import PupilCandidate


def draw_detection(
    image: np.ndarray,
    left_candidate: PupilCandidate,
    right_candidate: PupilCandidate,
    distance_px: float,
) -> np.ndarray:
    annotated = ensure_bgr(image)

    for label, candidate, color in (
        ("L", left_candidate, (0, 255, 0)),
        ("R", right_candidate, (0, 165, 255)),
    ):
        center = tuple(int(v) for v in candidate.center)
        radius = max(2, int(round(candidate.radius)))
        cv2.circle(annotated, center, radius, color, 2)
        cv2.circle(annotated, center, 2, (0, 0, 255), -1)
        if candidate.contour is not None:
            cv2.drawContours(annotated, [candidate.contour], -1, color, 1)
        if candidate.bbox is not None:
            x, y, w, h = candidate.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            annotated,
            f"{label}:{center}",
            (center[0] + 4, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.line(annotated, left_candidate.center, right_candidate.center, (255, 0, 0), 1)
    mid_x = (left_candidate.center[0] + right_candidate.center[0]) // 2
    mid_y = (left_candidate.center[1] + right_candidate.center[1]) // 2
    cv2.putText(
        annotated,
        f"Distance: {distance_px:.2f}px",
        (max(10, mid_x - 80), max(20, mid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated
