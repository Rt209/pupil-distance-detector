from __future__ import annotations

from typing import Iterable

import numpy as np

from pupil_distance_detector.utils.models import PupilCandidate


def _position_score(candidate: PupilCandidate, width: int, height: int) -> float:
    x, y = candidate.center
    horizontal_centering = 1.0 - abs((x / width) - 0.5)
    upper_face_bias = 1.0 - abs((y / height) - 0.4)
    return max(0.0, horizontal_centering * 0.4 + upper_face_bias * 0.6)


def rank_candidates(
    candidates: Iterable[PupilCandidate],
    image_shape: tuple[int, int],
) -> list[PupilCandidate]:
    height, width = image_shape[:2]
    ranked = []
    for candidate in candidates:
        combined_score = candidate.score * 0.7 + _position_score(candidate, width, height) * 0.3
        ranked.append(
            PupilCandidate(
                center=candidate.center,
                radius=candidate.radius,
                score=combined_score,
                contour=candidate.contour,
                bbox=candidate.bbox,
                source=candidate.source,
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def select_best_pair(
    candidates: Iterable[PupilCandidate],
    image_shape: tuple[int, int],
) -> tuple[PupilCandidate, PupilCandidate]:
    ranked = rank_candidates(candidates, image_shape)
    if len(ranked) < 2:
        raise ValueError("Unable to find two valid pupil candidates.")

    height, width = image_shape[:2]
    best_pair: tuple[PupilCandidate, PupilCandidate] | None = None
    best_score = -np.inf

    for i, first in enumerate(ranked):
        for second in ranked[i + 1 :]:
            (left, right) = (first, second) if first.center[0] <= second.center[0] else (second, first)
            dx = right.center[0] - left.center[0]
            dy = abs(right.center[1] - left.center[1])
            if dx < width * 0.1:
                continue
            horizontal_ratio = dx / max(width, 1)
            vertical_penalty = 1.0 - min(dy / max(height * 0.2, 1), 1.0)
            symmetry = 1.0 - min(abs(left.radius - right.radius) / max(max(left.radius, right.radius), 1.0), 1.0)
            pair_score = left.score + right.score + horizontal_ratio + vertical_penalty + symmetry
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (left, right)

    if best_pair is None:
        raise ValueError("Unable to select a plausible left/right pupil pair.")
    return best_pair
