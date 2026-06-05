from __future__ import annotations

import cv2
import numpy as np

from .config import BackgroundModelConfig


class GaussianMixtureBackgroundModel:
    def __init__(self, config: BackgroundModelConfig) -> None:
        self.config = config
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows,
        )
        self._kernel = self._build_kernel(config.morph_kernel)

    def annotate(self, frame, mask=None):
        if mask is None:
            mask = self.foreground_mask(frame)
        annotated_frame = frame.copy()

        if self.config.show_mask:
            self._overlay_mask(annotated_frame, mask)

        if self.config.draw_contours:
            contours = self._foreground_contours(mask)
            cv2.drawContours(
                annotated_frame,
                contours,
                contourIdx=-1,
                color=self.config.contour_color,
                thickness=self.config.contour_thickness,
            )

        return annotated_frame

    def foreground_mask(self, frame):
        raw_mask = self._subtractor.apply(frame, learningRate=self.config.learning_rate)
        _, mask = cv2.threshold(
            raw_mask,
            self.config.binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        if self._kernel is None:
            return mask

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

    def _foreground_contours(self, mask) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.config.min_area
        ]

    def _overlay_mask(self, frame, mask) -> None:
        foreground_pixels = mask > 0
        if not np.any(foreground_pixels):
            return

        overlay = np.empty_like(frame)
        overlay[:] = self.config.foreground_color
        alpha = self.config.mask_alpha
        frame[foreground_pixels] = cv2.addWeighted(
            frame[foreground_pixels],
            1.0 - alpha,
            overlay[foreground_pixels],
            alpha,
            0,
        )

    def _build_kernel(self, kernel_size: int):
        if kernel_size <= 1:
            return None
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
