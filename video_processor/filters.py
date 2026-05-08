from __future__ import annotations

import cv2

from .config import FilterConfig


def apply_median_filter(frame, kernel_size: int):
    return cv2.medianBlur(frame, kernel_size)


def apply_gaussian_filter(frame, kernel_size: tuple[int, int]):
    return cv2.GaussianBlur(frame, kernel_size, 0)


def apply_configured_filters(frame, config: FilterConfig):
    processed_frame = frame.copy()

    if config.use_median:
        processed_frame = apply_median_filter(processed_frame, config.median_kernel)

    if config.use_gaussian:
        processed_frame = apply_gaussian_filter(processed_frame, config.gaussian_kernel)

    return processed_frame
