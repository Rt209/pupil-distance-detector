from __future__ import annotations

import cv2
import numpy as np

from pupil_distance_detector.measurement.center_estimation import center_from_contour
from pupil_distance_detector.utils.models import PupilCandidate


def detect_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour_candidates(
    contours: list[np.ndarray],
    gray_image: np.ndarray,
    min_area: float = 20.0,
    max_area_ratio: float = 0.03,
) -> list[PupilCandidate]:
    image_area = gray_image.shape[0] * gray_image.shape[1]
    candidates: list[PupilCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > image_area * max_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.2:
            continue

        center = center_from_contour(contour)
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray_image[y : y + h, x : x + w]
        darkness = 255.0 - float(np.mean(roi))
        score = (darkness / 255.0) * 0.7 + circularity * 0.3
        radius = max(w, h) / 2.0
        candidates.append(
            PupilCandidate(
                center=center,
                radius=radius,
                score=score,
                contour=contour,
                bbox=(x, y, w, h),
                source="contour",
            )
        )

    return candidates
