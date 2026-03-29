from __future__ import annotations

import numpy as np

from pupil_distance_detector.edge_detection.canny import detect_edges_canny
from pupil_distance_detector.feature_detection.candidate_selection import select_best_pair
from pupil_distance_detector.feature_detection.contour import contour_candidates, detect_contours
from pupil_distance_detector.measurement.distance import euclidean_distance
from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.preprocessing.blur import apply_gaussian_blur
from pupil_distance_detector.preprocessing.morphology import apply_morphology
from pupil_distance_detector.utils.drawing import draw_detection
from pupil_distance_detector.utils.image_io import ensure_gray
from pupil_distance_detector.utils.models import DetectionResult, PipelineArtifacts


class PipelineV1(BasePipeline):
    name = "v1"

    def run(self, image: np.ndarray) -> DetectionResult:
        gray = ensure_gray(image)
        blurred = apply_gaussian_blur(gray, kernel_size=(7, 7), sigma_x=1.2)
        edges = detect_edges_canny(blurred, threshold1=25, threshold2=80)
        closed = apply_morphology(edges, operation="close", kernel_size=(5, 5), iterations=1)
        contours = detect_contours(closed)
        candidates = contour_candidates(contours, gray)
        left, right = select_best_pair(candidates, gray.shape)
        distance_px = euclidean_distance(left.center, right.center)
        artifacts = PipelineArtifacts(gray=gray, stages={"blurred": blurred, "edges": edges, "closed": closed})
        annotated = draw_detection(image, left, right, distance_px)
        return DetectionResult(
            pipeline_name=self.name,
            left_candidate=left,
            right_candidate=right,
            distance_px=distance_px,
            annotated_image=annotated,
            artifacts=artifacts,
        )
