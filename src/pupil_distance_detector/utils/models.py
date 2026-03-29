from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PupilCandidate:
    center: tuple[int, int]
    radius: float
    score: float
    contour: np.ndarray | None = None
    bbox: tuple[int, int, int, int] | None = None
    source: str = "unknown"


@dataclass(slots=True)
class PipelineArtifacts:
    gray: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionResult:
    pipeline_name: str
    left_candidate: PupilCandidate
    right_candidate: PupilCandidate
    distance_px: float
    annotated_image: np.ndarray
    artifacts: PipelineArtifacts
