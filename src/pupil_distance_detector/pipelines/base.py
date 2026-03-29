from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pupil_distance_detector.utils.models import DetectionResult


class BasePipeline(ABC):
    name: str

    @abstractmethod
    def run(self, image: np.ndarray) -> DetectionResult:
        raise NotImplementedError
