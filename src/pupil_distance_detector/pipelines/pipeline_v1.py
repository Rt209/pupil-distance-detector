"""
Pipeline V1 技術流程
1. 讀入輸入影像並轉為灰階影像。
2. 進行小角度旋轉搜尋，降低人臉微幅傾斜造成的偵測偏差。
3. 使用 Haar cascade 偵測臉部與眼睛位置。
4. 依據臉部與眼睛位置建立左右眼搜尋區域。
5. 對搜尋區域套用 Gaussian Blur，降低雜訊干擾。
6. 在左右眼區域內以自適應二值化、Canny、Hough Circle 與 contour 產生瞳孔候選。
7. 以暗區、圓形程度與區域位置評估候選品質，收斂出左右瞳孔中心。
8. 計算左右瞳孔中心的像素距離。
9. 只輸出最終成果圖，清楚標示瞳孔範圍與瞳孔中心距離。
10. 保留中間處理結果於 artifacts，供除錯或後續分析使用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pupil_distance_detector.measurement.distance import euclidean_distance
from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.preprocessing.binarization import apply_adaptive_threshold
from pupil_distance_detector.preprocessing.blur import apply_gaussian_blur
from pupil_distance_detector.utils.image_io import ensure_bgr, ensure_gray
from pupil_distance_detector.utils.models import DetectionResult, PipelineArtifacts, PupilCandidate


FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"


class PipelineV1(BasePipeline):
    name = "v1"

    ROTATION_ANGLES = (-10.0, -5.0, 0.0, 5.0, 10.0)
    BLUR_OPTIONS = (
        ((7, 7), 1.0),
        ((9, 9), 1.4),
    )

    @staticmethod
    def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _make_box_mask(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        masked = np.zeros_like(image)
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        return masked

    @staticmethod
    def _detect_face_and_eyes(
        gray: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
        face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        if len(faces) == 0:
            return None, []

        fx, fy, fw, fh = map(int, max(faces, key=lambda item: item[2] * item[3]))
        face_upper = gray[fy : fy + int(fh * 0.62), fx : fx + fw]
        eyes = eye_cascade.detectMultiScale(face_upper, scaleFactor=1.05, minNeighbors=6, minSize=(28, 28))

        eye_boxes: list[tuple[int, int, int, int]] = []
        for ex, ey, ew, eh in eyes:
            eye_boxes.append((fx + int(ex), fy + int(ey), int(ew), int(eh)))

        eye_boxes.sort(key=lambda item: item[2] * item[3], reverse=True)
        return (fx, fy, fw, fh), eye_boxes[:4]

    @staticmethod
    def _build_eye_search_regions(
        gray: np.ndarray,
        face_box: tuple[int, int, int, int] | None,
        eye_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
        height, width = gray.shape[:2]

        if face_box is None:
            face_box = (int(width * 0.18), int(height * 0.10), int(width * 0.64), int(height * 0.72))

        fx, fy, fw, fh = face_box
        coarse_box = (
            max(0, fx + int(fw * 0.10)),
            max(0, fy + int(fh * 0.16)),
            min(width, fx + int(fw * 0.90)),
            min(height, fy + int(fh * 0.48)),
        )

        def expand_eye(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            ex, ey, ew, eh = box
            x1 = max(0, ex - int(ew * 0.10))
            x2 = min(width, ex + ew + int(ew * 0.10))
            y1 = max(0, ey + int(eh * 0.12))
            y2 = min(height, ey + int(eh * 0.72))
            return x1, y1, x2, y2

        if len(eye_boxes) >= 2:
            sorted_eyes = sorted(eye_boxes, key=lambda item: item[0])
            left_raw = sorted_eyes[0]
            right_raw = sorted_eyes[-1]
            avg_h = int(round((left_raw[3] + right_raw[3]) / 2))
            avg_center_y = int(
                round(((left_raw[1] + left_raw[3] / 2) + (right_raw[1] + right_raw[3] / 2)) / 2)
            )

            left_eye_box = (left_raw[0], max(0, avg_center_y - avg_h // 2), left_raw[2], avg_h)
            right_eye_box = (right_raw[0], max(0, avg_center_y - avg_h // 2), right_raw[2], avg_h)
            return coarse_box, expand_eye(left_eye_box), expand_eye(right_eye_box)

        if len(eye_boxes) == 1:
            ex, ey, ew, eh = eye_boxes[0]
            face_center_x = fx + fw / 2.0
            eye_center_x = ex + ew / 2.0
            mirrored_center_x = int(round(2 * face_center_x - eye_center_x))
            mirrored_x = max(fx, min(int(mirrored_center_x - ew / 2.0), fx + fw - ew))
            mirrored_eye = (mirrored_x, ey, ew, eh)

            if eye_center_x <= face_center_x:
                return coarse_box, expand_eye((ex, ey, ew, eh)), expand_eye(mirrored_eye)
            return coarse_box, expand_eye(mirrored_eye), expand_eye((ex, ey, ew, eh))

        cx1, cy1, cx2, cy2 = coarse_box
        coarse_width = cx2 - cx1
        coarse_height = cy2 - cy1
        left_box = (
            cx1,
            cy1 + int(coarse_height * 0.10),
            cx1 + int(coarse_width * 0.48),
            cy1 + int(coarse_height * 0.60),
        )
        right_box = (
            cx1 + int(coarse_width * 0.52),
            cy1 + int(coarse_height * 0.10),
            cx2,
            cy1 + int(coarse_height * 0.60),
        )
        return coarse_box, left_box, right_box

    @staticmethod
    def _score_local_candidate(candidate: PupilCandidate, roi: np.ndarray) -> float:
        cx, cy = candidate.center
        radius = max(4, int(round(candidate.radius)))
        rx1 = max(0, cx - radius)
        ry1 = max(0, cy - radius)
        rx2 = min(roi.shape[1], cx + radius)
        ry2 = min(roi.shape[0], cy + radius)
        patch = roi[ry1:ry2, rx1:rx2]
        darkness = 255.0 - float(np.mean(patch)) if patch.size else 0.0
        center_penalty = abs(cx - roi.shape[1] / 2) / max(roi.shape[1] / 2, 1)
        vertical_penalty = abs(cy - roi.shape[0] * 0.42) / max(roi.shape[0] * 0.42, 1)
        return candidate.score + (darkness / 255.0) * 1.5 - center_penalty * 0.6 - vertical_penalty * 0.8

    @staticmethod
    def _refine_candidate_center(roi: np.ndarray, candidate: PupilCandidate) -> tuple[tuple[int, int], float]:
        cx, cy = candidate.center
        radius = max(6, int(round(candidate.radius * 1.6)))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(roi.shape[1], cx + radius + 1)
        y2 = min(roi.shape[0], cy + radius + 1)
        patch = roi[y1:y2, x1:x2]
        if patch.size == 0:
            return candidate.center, candidate.radius

        patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)
        darkness = 255.0 - patch_blur.astype(np.float32)
        threshold = np.percentile(darkness, 85)
        mask = darkness >= threshold
        if not np.any(mask):
            min_idx = np.unravel_index(np.argmin(patch_blur), patch_blur.shape)
            return (x1 + int(min_idx[1]), y1 + int(min_idx[0])), candidate.radius

        ys, xs = np.where(mask)
        weights = darkness[ys, xs]
        refined_x = int(round(np.average(xs, weights=weights))) + x1
        refined_y = int(round(np.average(ys, weights=weights))) + y1
        blob_area = float(mask.sum())
        refined_radius = max(3.0, min(candidate.radius, np.sqrt(blob_area / np.pi)))
        return (refined_x, refined_y), refined_radius

    @classmethod
    def _detect_pupil_in_eye_region(
        cls,
        gray: np.ndarray,
        eye_box: tuple[int, int, int, int],
    ) -> tuple[PupilCandidate | None, dict[str, np.ndarray]]:
        x1, y1, x2, y2 = eye_box
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 12:
            return None, {}

        blurred = apply_gaussian_blur(roi, kernel_size=(7, 7), sigma_x=1.2)
        binary = apply_adaptive_threshold(blurred, block_size=25, c=7, invert=True)
        edges = cv2.Canny(blurred, 20, 60)
        candidates: list[PupilCandidate] = []

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(roi.shape[1] // 3, 10),
            param1=80,
            param2=8,
            minRadius=3,
            maxRadius=max(6, min(roi.shape[:2]) // 4),
        )
        if circles is not None:
            for cx, cy, radius in np.round(circles[0]).astype(int):
                if not (roi.shape[1] * 0.18 <= cx <= roi.shape[1] * 0.82):
                    continue
                if not (roi.shape[0] * 0.20 <= cy <= roi.shape[0] * 0.62):
                    continue
                candidates.append(
                    PupilCandidate(
                        center=(int(cx), int(cy)),
                        radius=float(radius),
                        score=1.0 - min(radius / max(min(roi.shape[:2]) // 3, 1), 1.0),
                        contour=None,
                        bbox=(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2)),
                        source="hough",
                    )
                )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or area > roi.shape[0] * roi.shape[1] * 0.18:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.18:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            if not (roi.shape[1] * 0.18 <= cx <= roi.shape[1] * 0.82):
                continue
            if not (roi.shape[0] * 0.20 <= cy <= roi.shape[0] * 0.62):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(
                PupilCandidate(
                    center=(cx, cy),
                    radius=max(w, h) / 2.0,
                    score=circularity,
                    contour=contour,
                    bbox=(x, y, w, h),
                    source="contour",
                )
            )

        if not candidates:
            min_idx = np.unravel_index(np.argmin(blurred), blurred.shape)
            candidates.append(
                PupilCandidate(
                    center=(int(min_idx[1]), int(min_idx[0])),
                    radius=float(max(4, min(roi.shape[:2]) // 8)),
                    score=0.1,
                    contour=None,
                    bbox=None,
                    source="darkest-point",
                )
            )

        best_local = max(candidates, key=lambda item: cls._score_local_candidate(item, roi))
        refined_center, refined_radius = cls._refine_candidate_center(blurred, best_local)
        best_candidate = PupilCandidate(
            center=(x1 + refined_center[0], y1 + refined_center[1]),
            radius=refined_radius,
            score=cls._score_local_candidate(best_local, roi),
            contour=best_local.contour,
            bbox=(
                x1 + best_local.bbox[0],
                y1 + best_local.bbox[1],
                best_local.bbox[2],
                best_local.bbox[3],
            )
            if best_local.bbox
            else None,
            source=best_local.source,
        )
        return best_candidate, {"blurred": blurred, "binary": binary, "edges": edges}

    @staticmethod
    def _score_pair(
        left_candidate: PupilCandidate,
        right_candidate: PupilCandidate,
        image_shape: tuple[int, int],
        rotation_angle: float,
    ) -> float:
        height, width = image_shape[:2]
        dx = abs(right_candidate.center[0] - left_candidate.center[0])
        dy = abs(right_candidate.center[1] - left_candidate.center[1])
        horizontal_score = min(dx / max(width * 0.18, 1), 1.4)
        vertical_score = 1.0 - min(dy / max(height * 0.12, 1), 1.0)
        radius_gap = abs(left_candidate.radius - right_candidate.radius)
        radius_score = 1.0 - min(radius_gap / max(max(left_candidate.radius, right_candidate.radius), 1.0), 1.0)
        eye_line_penalty = abs(dx - width * 0.28) / max(width * 0.28, 1)
        angle_penalty = 0.02 * abs(rotation_angle)
        return (
            left_candidate.score
            + right_candidate.score
            + horizontal_score
            + vertical_score
            + radius_score
            - eye_line_penalty
            - angle_penalty
        )

    @classmethod
    def _auto_search_pupils(cls, gray: np.ndarray) -> dict[str, object]:
        best_result: dict[str, object] | None = None
        best_score = -1.0

        for rotation_angle in cls.ROTATION_ANGLES:
            rotated_gray = cls._rotate_image(gray, rotation_angle)
            face_box, eye_boxes = cls._detect_face_and_eyes(rotated_gray)
            coarse_box, left_eye_box, right_eye_box = cls._build_eye_search_regions(rotated_gray, face_box, eye_boxes)
            coarse_roi_masked = cls._make_box_mask(rotated_gray, coarse_box)
            left_eye_masked = cls._make_box_mask(rotated_gray, left_eye_box)
            right_eye_masked = cls._make_box_mask(rotated_gray, right_eye_box)

            for blur_kernel, blur_sigma in cls.BLUR_OPTIONS:
                blurred = apply_gaussian_blur(rotated_gray, kernel_size=blur_kernel, sigma_x=blur_sigma)
                left_candidate, left_debug = cls._detect_pupil_in_eye_region(blurred, left_eye_box)
                right_candidate, right_debug = cls._detect_pupil_in_eye_region(blurred, right_eye_box)
                if left_candidate is None or right_candidate is None:
                    continue

                pair_score = cls._score_pair(left_candidate, right_candidate, rotated_gray.shape, rotation_angle)
                if pair_score <= best_score:
                    continue

                best_score = pair_score
                best_result = {
                    "rotated_gray": rotated_gray,
                    "rotation_angle": rotation_angle,
                    "blurred": blurred,
                    "face_box": face_box,
                    "eye_boxes": eye_boxes,
                    "coarse_roi_masked": coarse_roi_masked,
                    "coarse_roi_box": coarse_box,
                    "left_eye_masked": left_eye_masked,
                    "left_eye_box": left_eye_box,
                    "right_eye_masked": right_eye_masked,
                    "right_eye_box": right_eye_box,
                    "left_debug": left_debug,
                    "right_debug": right_debug,
                    "left_candidate": left_candidate,
                    "right_candidate": right_candidate,
                    "pair_source": "eye-region-local-search",
                    "distance_px": euclidean_distance(left_candidate.center, right_candidate.center),
                    "params": {"blur_kernel": blur_kernel, "blur_sigma": blur_sigma},
                }

        if best_result is None:
            raise ValueError("找不到穩定的瞳孔位置，請嘗試更換輸入影像或調整偵測策略。")
        return best_result

    @staticmethod
    def _annotate_result(
        base_image: np.ndarray,
        left_candidate: PupilCandidate,
        right_candidate: PupilCandidate,
        distance_px: float,
    ) -> np.ndarray:
        annotated = ensure_bgr(base_image)

        for candidate, color in (
            (left_candidate, (255, 255, 255)),
            (right_candidate, (255, 255, 255)),
        ):
            center = tuple(int(v) for v in candidate.center)
            radius = max(4, int(round(candidate.radius)))
            cv2.circle(annotated, center, radius, color, 2)
            cv2.circle(annotated, center, 3, (255, 255, 255), -1)

        text = f"PD: {distance_px:.2f} px"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_x = max(12, annotated.shape[1] - text_width - 24)
        label_y = max(text_height + 18, annotated.shape[0] - 18)

        cv2.rectangle(
            annotated,
            (label_x - 10, label_y - text_height - 10),
            (label_x + text_width + 10, label_y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (label_x, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return annotated

    def run(self, image: np.ndarray) -> DetectionResult:
        gray = ensure_gray(image)
        search_result = self._auto_search_pupils(gray)

        rotated_gray = search_result["rotated_gray"]
        blurred = search_result["blurred"]
        coarse_roi_masked = search_result["coarse_roi_masked"]
        left_eye_masked = search_result["left_eye_masked"]
        right_eye_masked = search_result["right_eye_masked"]
        left_debug = search_result["left_debug"]
        right_debug = search_result["right_debug"]
        left_candidate = search_result["left_candidate"]
        right_candidate = search_result["right_candidate"]
        distance_px = search_result["distance_px"]

        annotated = self._annotate_result(rotated_gray, left_candidate, right_candidate, distance_px)
        artifacts = PipelineArtifacts(
            gray=gray,
            stages={
                "rotated_gray": rotated_gray,
                "blurred": blurred,
                "coarse_roi_masked": coarse_roi_masked,
                "left_eye_masked": left_eye_masked,
                "right_eye_masked": right_eye_masked,
                "left_binary": left_debug["binary"],
                "right_binary": right_debug["binary"],
                "left_edges": left_debug["edges"],
                "right_edges": right_debug["edges"],
            },
            metadata={
                "rotation_angle": search_result["rotation_angle"],
                "pair_source": search_result["pair_source"],
                "params": search_result["params"],
                "face_box": search_result["face_box"],
                "eye_boxes": search_result["eye_boxes"],
                "left_eye_box": search_result["left_eye_box"],
                "right_eye_box": search_result["right_eye_box"],
            },
        )

        return DetectionResult(
            pipeline_name=self.name,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            distance_px=distance_px,
            annotated_image=annotated,
            artifacts=artifacts,
        )
