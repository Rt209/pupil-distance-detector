from __future__ import annotations

import numpy as np

from pupil_distance_detector.feature_detection.candidate_selection import select_best_pair
from pupil_distance_detector.feature_detection.hough_circle import detect_hough_circles
from pupil_distance_detector.measurement.distance import euclidean_distance
from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.preprocessing.binarization import apply_adaptive_threshold
from pupil_distance_detector.preprocessing.blur import apply_gaussian_blur
from pupil_distance_detector.preprocessing.morphology import apply_morphology
from pupil_distance_detector.utils.drawing import draw_detection
from pupil_distance_detector.utils.image_io import ensure_gray
from pupil_distance_detector.utils.models import DetectionResult, PipelineArtifacts


class PipelineV2(BasePipeline):
    name = "v2"

    def run(self, image: np.ndarray) -> DetectionResult:
        gray = ensure_gray(image)
        blurred = apply_gaussian_blur(gray, kernel_size=(9, 9), sigma_x=1.5)
        binary = apply_adaptive_threshold(blurred, block_size=31, c=7, invert=True)
        opened = apply_morphology(binary, operation="open", kernel_size=(3, 3), iterations=1)
        closed = apply_morphology(opened, operation="close", kernel_size=(5, 5), iterations=2)
        circles = detect_hough_circles(closed, dp=1.2, min_dist=20.0, param1=80.0, param2=10.0)
        left, right = select_best_pair(circles, gray.shape)
        distance_px = euclidean_distance(left.center, right.center)
        artifacts = PipelineArtifacts(gray=gray, stages={"blurred": blurred, "binary": binary, "opened": opened, "closed": closed})
        annotated = draw_detection(image, left, right, distance_px)
        return DetectionResult(
            pipeline_name=self.name,
            left_candidate=left,
            right_candidate=right,
            distance_px=distance_px,
            annotated_image=annotated,
            artifacts=artifacts,
        )
