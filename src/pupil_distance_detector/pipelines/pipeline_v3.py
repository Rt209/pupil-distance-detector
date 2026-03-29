from __future__ import annotations

import numpy as np

from pupil_distance_detector.edge_detection.sobel import detect_edges_sobel
from pupil_distance_detector.feature_detection.candidate_selection import select_best_pair
from pupil_distance_detector.feature_detection.contour import contour_candidates, detect_contours
from pupil_distance_detector.measurement.distance import euclidean_distance
from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.preprocessing.blur import apply_gaussian_blur
from pupil_distance_detector.preprocessing.binarization import apply_binary_threshold
from pupil_distance_detector.preprocessing.morphology import apply_morphology
from pupil_distance_detector.preprocessing.perspective import apply_perspective_transform
from pupil_distance_detector.utils.drawing import draw_detection
from pupil_distance_detector.utils.image_io import ensure_gray
from pupil_distance_detector.utils.models import DetectionResult, PipelineArtifacts


class PipelineV3(BasePipeline):
    name = "v3"

    def run(self, image: np.ndarray) -> DetectionResult:
        gray = ensure_gray(image)
        rectified = apply_perspective_transform(gray)
        blurred = apply_gaussian_blur(rectified, kernel_size=(5, 5), sigma_x=1.0)
        sobel = detect_edges_sobel(blurred, kernel_size=3)
        binary = apply_binary_threshold(sobel, threshold=40, invert=False)
        closed = apply_morphology(binary, operation="close", kernel_size=(5, 5), iterations=1)
        dilated = apply_morphology(closed, operation="dilate", kernel_size=(3, 3), iterations=1)
        contours = detect_contours(dilated)
        candidates = contour_candidates(contours, rectified)
        left, right = select_best_pair(candidates, rectified.shape)
        distance_px = euclidean_distance(left.center, right.center)
        artifacts = PipelineArtifacts(
            gray=gray,
            stages={
                "rectified": rectified,
                "blurred": blurred,
                "sobel": sobel,
                "binary": binary,
                "closed": closed,
                "dilated": dilated,
            },
        )
        annotated = draw_detection(rectified, left, right, distance_px)
        return DetectionResult(
            pipeline_name=self.name,
            left_candidate=left,
            right_candidate=right,
            distance_px=distance_px,
            annotated_image=annotated,
            artifacts=artifacts,
        )
