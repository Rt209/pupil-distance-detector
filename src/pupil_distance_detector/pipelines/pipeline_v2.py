"""
Pipeline V2 技術流程
1. 影像轉為灰階影像。
2. 小角度旋轉搜尋，降低人臉傾斜造成的偏差。
3. 使用 Haar cascade 偵測臉部位置。
4. 依據臉部比例建立四個 reference points。
5. 使用 reference points 做 Perspective Transform，拉正眼帶區域。
6. 在校正後的眼帶區域切出左眼與右眼 ROI。
7. 在每個眼睛 ROI 內依序進行：
   - gradient voting 尋找 iris / pupil 粗略中心
   - dark-center refinement 微調中心
   - radial intensity profile 估計瞳孔半徑
   - Hough circle 進一步微調中心與半徑
8. 將左右瞳孔中心與輪廓反投影回原始影像座標。
9. 計算左右瞳孔中心距離，並輸出最終標註成果圖。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pupil_distance_detector.measurement.distance import euclidean_distance
from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.utils.image_io import ensure_bgr, ensure_gray
from pupil_distance_detector.utils.models import DetectionResult, PipelineArtifacts, PupilCandidate


FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class PipelineV2(BasePipeline):
    name = "v2"

    ROTATION_ANGLES = (-6.0, -3.0, 0.0, 3.0, 6.0)
    REFERENCE_POINT_SETS: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.14, 0.22), (0.86, 0.20), (0.84, 0.42), (0.16, 0.44)),
        ((0.13, 0.21), (0.87, 0.19), (0.85, 0.41), (0.15, 0.43)),
        ((0.15, 0.23), (0.85, 0.21), (0.83, 0.43), (0.17, 0.45)),
    )
    WARP_W = 480
    WARP_H = 180

    @staticmethod
    def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _detect_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
        cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return tuple(map(int, max(faces, key=lambda item: item[2] * item[3])))

    @staticmethod
    def _build_src_points(
        gray: np.ndarray,
        face_box: tuple[int, int, int, int] | None,
        ratios: tuple[tuple[float, float], ...],
    ) -> np.ndarray:
        height, width = gray.shape[:2]
        if face_box is None:
            face_box = (int(width * 0.10), int(height * 0.06), int(width * 0.80), int(height * 0.84))
        fx, fy, fw, fh = face_box
        points = [[fx + fw * rx, fy + fh * ry] for rx, ry in ratios]
        return np.float32(points)

    @classmethod
    def _warp_eye_band(
        cls,
        gray: np.ndarray,
        src_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dst_points = np.float32(
            [
                [0, 0],
                [cls.WARP_W - 1, 0],
                [cls.WARP_W - 1, cls.WARP_H - 1],
                [0, cls.WARP_H - 1],
            ]
        )
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        warped = cv2.warpPerspective(gray, perspective_matrix, (cls.WARP_W, cls.WARP_H))
        return warped, perspective_matrix, inverse_matrix

    @staticmethod
    def _get_eye_boxes(warped_shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        height, width = warped_shape
        left_box = (int(width * 0.03), int(height * 0.25), int(width * 0.46), int(height * 0.95))
        right_box = (int(width * 0.54), int(height * 0.25), int(width * 0.97), int(height * 0.95))
        return left_box, right_box

    @staticmethod
    def _gradient_voting_fast(
        gray_small: np.ndarray,
        sigma: float = 1.5,
        grad_percentile: float = 70.0,
        step: int = 1,
    ) -> np.ndarray:
        height, width = gray_small.shape
        blurred = cv2.GaussianBlur(gray_small, (0, 0), sigma)

        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8

        threshold = np.percentile(magnitude, grad_percentile)
        mask = magnitude >= threshold

        grad_x_norm = grad_x / magnitude
        grad_y_norm = grad_y / magnitude
        weight = (255.0 - blurred.astype(np.float64)) / 255.0

        ys, xs = np.where(mask)
        grad_x_sel = grad_x_norm[ys, xs]
        grad_y_sel = grad_y_norm[ys, xs]
        weight_sel = weight[ys, xs] * magnitude[ys, xs] / np.max(magnitude)
        objective_map = np.zeros((height, width), dtype=np.float64)

        for cy in range(0, height, step):
            dy = ys.astype(np.float64) - cy
            for cx in range(0, width, step):
                dx = xs.astype(np.float64) - cx
                dist = np.sqrt(dx**2 + dy**2) + 1e-8
                dot = (dx * grad_x_sel + dy * grad_y_sel) / dist
                positive = np.maximum(dot, 0.0)
                objective_map[cy, cx] = np.sum(positive**2 * weight_sel)

        return objective_map

    @classmethod
    def _gradient_voting_center(
        cls,
        roi_gray: np.ndarray,
        search_frac: float = 0.6,
    ) -> tuple[int, int, float]:
        height, width = roi_gray.shape
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

        target_size = 30
        scale1 = min(1.0, target_size / max(height, width, 1))
        small_h = max(10, int(height * scale1))
        small_w = max(10, int(width * scale1))
        small1 = cv2.resize(roi_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small1 = clahe.apply(small1)

        objective1 = cls._gradient_voting_fast(small1, sigma=1.2, grad_percentile=65, step=1)
        margin_x = int(small_w * (1 - search_frac) / 2)
        margin_y = int(small_h * (1 - search_frac) / 2)
        objective1[:margin_y, :] = 0
        objective1[small_h - max(margin_y, 1) :, :] = 0
        objective1[:, :margin_x] = 0
        objective1[:, small_w - max(margin_x, 1) :] = 0
        objective1 = cv2.GaussianBlur(objective1, (3, 3), 0.8)
        _, _, _, max_loc1 = cv2.minMaxLoc(objective1)

        coarse_x = int(round(max_loc1[0] / scale1))
        coarse_y = int(round(max_loc1[1] / scale1))

        refine_radius = max(12, int(min(height, width) * 0.20))
        x1 = max(0, coarse_x - refine_radius)
        y1 = max(0, coarse_y - refine_radius)
        x2 = min(width, coarse_x + refine_radius)
        y2 = min(height, coarse_y + refine_radius)
        patch = roi_gray[y1:y2, x1:x2]

        if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
            coarse_x = min(max(coarse_x, 0), width - 1)
            coarse_y = min(max(coarse_y, 0), height - 1)
            return coarse_x, coarse_y, 0.5

        scale2 = min(1.0, 35.0 / max(patch.shape[0], patch.shape[1], 1))
        patch_h = max(8, int(patch.shape[0] * scale2))
        patch_w = max(8, int(patch.shape[1] * scale2))
        small2 = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        small2 = clahe.apply(small2)

        objective2 = cls._gradient_voting_fast(small2, sigma=1.0, grad_percentile=60, step=1)
        objective2 = cv2.GaussianBlur(objective2, (3, 3), 0.5)
        max_val, _, _, max_loc2 = cv2.minMaxLoc(objective2)

        fine_x = int(round(max_loc2[0] / scale2)) + x1
        fine_y = int(round(max_loc2[1] / scale2)) + y1
        fine_x = min(max(fine_x, 0), width - 1)
        fine_y = min(max(fine_y, 0), height - 1)
        confidence = min(1.0, max_val / max(np.mean(objective2) + 1e-8, 1.0))
        return fine_x, fine_y, confidence

    @staticmethod
    def _dark_center_refine(
        roi_gray: np.ndarray,
        cx: int,
        cy: int,
        search_radius: int = 15,
    ) -> tuple[int, int]:
        height, width = roi_gray.shape
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(width, cx + search_radius)
        y2 = min(height, cy + search_radius)
        patch = roi_gray[y1:y2, x1:x2]
        if patch.size == 0:
            return cx, cy

        blurred = cv2.GaussianBlur(patch, (7, 7), 2.0)
        _, _, min_loc, _ = cv2.minMaxLoc(blurred)
        dark_x = min_loc[0] + x1
        dark_y = min_loc[1] + y1
        dist = np.hypot(dark_x - cx, dark_y - cy)
        if dist > search_radius * 0.8:
            return cx, cy

        final_x = int(round(cx * 0.4 + dark_x * 0.6))
        final_y = int(round(cy * 0.4 + dark_y * 0.6))
        return min(max(final_x, 0), width - 1), min(max(final_y, 0), height - 1)

    @staticmethod
    def _estimate_pupil_radius(
        roi_gray: np.ndarray,
        cx: int,
        cy: int,
        min_r: int = 3,
        max_r: int = 40,
    ) -> int:
        height, width = roi_gray.shape
        max_r = min(max_r, min(cx, cy, width - cx - 1, height - cy - 1, int(min(height, width) * 0.45)))
        if max_r <= min_r + 2:
            return max(min_r, 5)

        radial_profile = []
        for radius in range(min_r, max_r + 1):
            num_points = max(12, int(2 * np.pi * radius))
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            xs = cx + (radius * np.cos(angles)).astype(int)
            ys = cy + (radius * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if valid.sum() < 4:
                radial_profile.append(radial_profile[-1] if radial_profile else 128.0)
                continue
            radial_profile.append(float(np.mean(roi_gray[ys[valid], xs[valid]])))

        profile = np.array(radial_profile)
        if len(profile) < 3:
            return max(min_r, 5)

        gradient = np.diff(profile)
        if len(gradient) >= 5:
            gradient = np.convolve(gradient, np.ones(3) / 3, mode="same")

        best_idx = int(np.argmax(gradient))
        pupil_r = best_idx + min_r
        return max(min_r + 1, pupil_r)

    @staticmethod
    def _hough_refine_pupil(
        roi_gray: np.ndarray,
        cx_init: int,
        cy_init: int,
        r_init: int,
        search_radius: int = 20,
    ) -> tuple[int, int, int]:
        height, width = roi_gray.shape
        patch_radius = search_radius * 2
        x1 = max(0, cx_init - patch_radius)
        y1 = max(0, cy_init - patch_radius)
        x2 = min(width, cx_init + patch_radius)
        y2 = min(height, cy_init + patch_radius)
        patch = roi_gray[y1:y2, x1:x2]

        if patch.size == 0 or patch.shape[0] < 10 or patch.shape[1] < 10:
            return cx_init, cy_init, r_init

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(patch)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

        min_r = max(3, r_init - 8)
        max_r = min(r_init + 15, min(patch.shape) // 2)
        if min_r >= max_r:
            return cx_init, cy_init, r_init

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=max(10, r_init),
            param1=80,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return cx_init, cy_init, r_init

        local_x = cx_init - x1
        local_y = cy_init - y1
        best_dist = float("inf")
        best = (cx_init, cy_init, r_init)
        for circle in circles[0]:
            hcx, hcy, hr = float(circle[0]), float(circle[1]), float(circle[2])
            dist = np.hypot(hcx - local_x, hcy - local_y)
            if dist < best_dist and dist < search_radius:
                best_dist = dist
                best = (int(round(hcx)) + x1, int(round(hcy)) + y1, int(round(hr)))
        return best

    @classmethod
    def _detect_pupil_in_roi(
        cls,
        warped_gray: np.ndarray,
        eye_box: tuple[int, int, int, int],
    ) -> PupilCandidate | None:
        x1, y1, x2, y2 = eye_box
        roi = warped_gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
            return None

        roi_h, roi_w = roi.shape
        grad_x, grad_y, confidence = cls._gradient_voting_center(roi, search_frac=0.60)
        search_r = max(8, int(min(roi_h, roi_w) * 0.15))
        pupil_x, pupil_y = cls._dark_center_refine(roi, grad_x, grad_y, search_radius=search_r)

        max_r_limit = max(5, int(min(roi_h, roi_w) * 0.35))
        pupil_r = cls._estimate_pupil_radius(roi, pupil_x, pupil_y, min_r=3, max_r=max_r_limit)

        hough_x, hough_y, hough_r = cls._hough_refine_pupil(
            roi,
            pupil_x,
            pupil_y,
            pupil_r,
            search_radius=max(12, pupil_r + 5),
        )

        hough_dist = np.hypot(hough_x - pupil_x, hough_y - pupil_y)
        if hough_dist < pupil_r * 1.5:
            final_x = int(round(pupil_x * 0.5 + hough_x * 0.5))
            final_y = int(round(pupil_y * 0.5 + hough_y * 0.5))
            final_r = int(round(pupil_r * 0.4 + hough_r * 0.6))
        else:
            final_x, final_y, final_r = pupil_x, pupil_y, pupil_r

        final_r = max(3, final_r)
        num_points = 64
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        contour_pts = np.zeros((num_points, 1, 2), dtype=np.int32)
        contour_pts[:, 0, 0] = (final_x + final_r * np.cos(angles)).astype(int)
        contour_pts[:, 0, 1] = (final_y + final_r * np.sin(angles)).astype(int)

        shifted_contour = contour_pts.copy()
        shifted_contour[:, 0, 0] += x1
        shifted_contour[:, 0, 1] += y1

        return PupilCandidate(
            center=(x1 + final_x, y1 + final_y),
            radius=float(final_r),
            score=confidence,
            contour=shifted_contour,
            source="gradient-voting-radial-hough",
        )

    @staticmethod
    def _transform_point_back(point: tuple[int, int], inverse_matrix: np.ndarray) -> tuple[int, int]:
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src, inverse_matrix)[0][0]
        return int(round(mapped[0])), int(round(mapped[1]))

    @staticmethod
    def _transform_contour_back(contour: np.ndarray, inverse_matrix: np.ndarray) -> np.ndarray:
        mapped = cv2.perspectiveTransform(contour.astype(np.float32), inverse_matrix)
        return np.round(mapped).astype(np.int32)

    @staticmethod
    def _score_pair(
        left_candidate: PupilCandidate,
        right_candidate: PupilCandidate,
        image_shape: tuple[int, int],
        angle: float,
    ) -> float:
        height, width = image_shape[:2]
        dx = abs(right_candidate.center[0] - left_candidate.center[0])
        dy = abs(right_candidate.center[1] - left_candidate.center[1])
        horizontal_score = min(dx / max(width * 0.12, 1), 1.8)
        vertical_score = 1.0 - min(dy / max(height * 0.10, 1), 1.0)
        radius_gap = abs(left_candidate.radius - right_candidate.radius) / max(
            max(left_candidate.radius, right_candidate.radius),
            1.0,
        )
        radius_score = 1.0 - min(radius_gap, 1.0)
        angle_penalty = 0.04 * abs(angle)
        return (
            left_candidate.score
            + right_candidate.score
            + horizontal_score
            + vertical_score
            + radius_score
            - angle_penalty
        )

    @classmethod
    def _auto_search_pupils(cls, image: np.ndarray) -> dict[str, object]:
        gray = ensure_gray(image)
        best_result: dict[str, object] | None = None
        best_score = -1.0

        for angle in cls.ROTATION_ANGLES:
            rotated_gray = cls._rotate_image(gray, angle)
            face_box = cls._detect_face_box(rotated_gray)

            for ratios in cls.REFERENCE_POINT_SETS:
                src_points = cls._build_src_points(rotated_gray, face_box, ratios)
                warped_gray, perspective_matrix, inverse_matrix = cls._warp_eye_band(rotated_gray, src_points)
                left_box, right_box = cls._get_eye_boxes(warped_gray.shape)

                left_warp = cls._detect_pupil_in_roi(warped_gray, left_box)
                right_warp = cls._detect_pupil_in_roi(warped_gray, right_box)
                if left_warp is None or right_warp is None:
                    continue

                left_center = cls._transform_point_back(left_warp.center, inverse_matrix)
                right_center = cls._transform_point_back(right_warp.center, inverse_matrix)
                left_contour = cls._transform_contour_back(left_warp.contour, inverse_matrix) if left_warp.contour is not None else None
                right_contour = cls._transform_contour_back(right_warp.contour, inverse_matrix) if right_warp.contour is not None else None

                left_candidate = PupilCandidate(
                    center=left_center,
                    radius=left_warp.radius,
                    score=left_warp.score,
                    contour=left_contour,
                    source=left_warp.source,
                )
                right_candidate = PupilCandidate(
                    center=right_center,
                    radius=right_warp.radius,
                    score=right_warp.score,
                    contour=right_contour,
                    source=right_warp.source,
                )

                pair_score = cls._score_pair(left_candidate, right_candidate, rotated_gray.shape, angle)
                if pair_score <= best_score:
                    continue

                best_score = pair_score
                best_result = {
                    "rotated_gray": rotated_gray,
                    "rotation_angle": angle,
                    "face_box": face_box,
                    "src_points": src_points,
                    "inverse_matrix": inverse_matrix,
                    "warped_gray": warped_gray,
                    "left_box": left_box,
                    "right_box": right_box,
                    "left_candidate": left_candidate,
                    "right_candidate": right_candidate,
                    "distance_px": euclidean_distance(left_center, right_center),
                    "pair_score": pair_score,
                    "params": {"angle": angle, "ratios": ratios},
                }

        if best_result is None:
            raise ValueError("找不到可用的瞳孔偵測結果，請嘗試更換輸入影像。")
        return best_result

    @staticmethod
    def _annotate_result(
        base_image: np.ndarray,
        left_candidate: PupilCandidate,
        right_candidate: PupilCandidate,
        distance_px: float,
    ) -> np.ndarray:
        annotated = ensure_bgr(base_image).copy()
        scale = max(annotated.shape[0], annotated.shape[1]) / 1200.0
        outline_t = max(2, int(round(scale * 3.0)))
        inner_t = max(1, int(round(scale * 1.8)))
        dot_r = max(3, int(round(scale * 2.5)))

        for candidate in (left_candidate, right_candidate):
            center = tuple(int(v) for v in candidate.center)
            if candidate.contour is not None and len(candidate.contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(candidate.contour)
                    cv2.ellipse(annotated, ellipse, (0, 0, 0), outline_t, cv2.LINE_AA)
                    cv2.ellipse(annotated, ellipse, (255, 255, 255), inner_t, cv2.LINE_AA)
                except cv2.error:
                    cv2.polylines(annotated, [candidate.contour], True, (0, 0, 0), outline_t, cv2.LINE_AA)
                    cv2.polylines(annotated, [candidate.contour], True, (255, 255, 255), inner_t, cv2.LINE_AA)
            else:
                radius = max(6, int(round(candidate.radius * 2.0)))
                cv2.circle(annotated, center, radius, (0, 0, 0), outline_t, cv2.LINE_AA)
                cv2.circle(annotated, center, radius, (255, 255, 255), inner_t, cv2.LINE_AA)

            cv2.circle(annotated, center, dot_r + 1, (0, 0, 0), -1)
            cv2.circle(annotated, center, dot_r, (255, 255, 255), -1)

        text = f"PD: {distance_px:.2f} px"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, scale * 0.9)
        thickness = max(1, int(round(scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_x = max(12, annotated.shape[1] - text_width - 20)
        label_y = max(text_height + 12, annotated.shape[0] - 16)
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
        search_result = self._auto_search_pupils(image)
        left_candidate = search_result["left_candidate"]
        right_candidate = search_result["right_candidate"]
        distance_px = search_result["distance_px"]
        rotated_gray = search_result["rotated_gray"]
        warped_gray = search_result["warped_gray"]

        artifacts = PipelineArtifacts(
            gray=gray,
            stages={
                "rotated_gray": rotated_gray,
                "warped_gray": warped_gray,
            },
            metadata={
                "rotation_angle": search_result["rotation_angle"],
                "face_box": search_result["face_box"],
                "src_points": search_result["src_points"].tolist(),
                "left_box": search_result["left_box"],
                "right_box": search_result["right_box"],
                "pair_score": search_result["pair_score"],
                "params": search_result["params"],
            },
        )
        annotated = self._annotate_result(rotated_gray, left_candidate, right_candidate, distance_px)
        return DetectionResult(
            pipeline_name=self.name,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            distance_px=distance_px,
            annotated_image=annotated,
            artifacts=artifacts,
        )
