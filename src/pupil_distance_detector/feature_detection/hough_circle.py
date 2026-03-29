from __future__ import annotations

import cv2
import numpy as np

from pupil_distance_detector.utils.models import PupilCandidate


def detect_hough_circles(
    image: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 20.0,
    param1: float = 80.0,
    param2: float = 12.0,
    min_radius: int = 3,
    max_radius: int = 40,
) -> list[PupilCandidate]:
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    candidates: list[PupilCandidate] = []
    for x, y, radius in np.round(circles[0]).astype(int):
        score = max(0.0, 1.0 - (radius / max(max_radius, 1)))
        candidates.append(
            PupilCandidate(
                center=(int(x), int(y)),
                radius=float(radius),
                score=score,
                contour=None,
                bbox=(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2)),
                source="hough",
            )
        )
    return candidates
