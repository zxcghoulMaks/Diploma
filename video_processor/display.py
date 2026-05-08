from __future__ import annotations

import cv2
import numpy as np


class DisplayWindow:
    def __init__(self, name: str, frame_size: tuple[int, int], normal_max_height: int) -> None:
        self.name = name
        self.frame_width, self.frame_height = frame_size
        self.normal_max_height = normal_max_height
        self.fullscreen_mode = False

        normal_scale = self.normal_max_height / self.frame_height
        self.normal_width = int(self.frame_width * normal_scale)
        self.normal_height = int(self.frame_height * normal_scale)

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.normal_width, self.normal_height)

    def show(self, frame) -> bool:
        if not self.is_open():
            return False

        display_frame = self._build_display_frame(frame)
        cv2.imshow(self.name, display_frame)
        return True

    def is_open(self) -> bool:
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def poll_key(self, delay_ms: int = 25) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def toggle_fullscreen(self) -> None:
        self.fullscreen_mode = not self.fullscreen_mode

        if self.fullscreen_mode:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return

        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.normal_width, self.normal_height)

    def _build_display_frame(self, frame):
        if not self.fullscreen_mode:
            return cv2.resize(
                frame,
                (self.normal_width, self.normal_height),
                interpolation=cv2.INTER_AREA,
            )

        frame_height, frame_width = frame.shape[:2]
        window_width, window_height = self._read_window_size(frame_width, frame_height)
        scale = min(window_width / frame_width, window_height / frame_height)

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        resized_frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2
        display_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        return display_frame

    def _read_window_size(self, fallback_width: int, fallback_height: int) -> tuple[int, int]:
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(self.name)
        except cv2.error:
            return fallback_width, fallback_height

        if window_width <= 0 or window_height <= 0:
            return fallback_width, fallback_height

        return window_width, window_height
