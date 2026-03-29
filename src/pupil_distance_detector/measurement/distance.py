from __future__ import annotations

import math


def euclidean_distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.dist(point_a, point_b)
