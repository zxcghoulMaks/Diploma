from __future__ import annotations

from pathlib import Path

import cv2


def open_video_source(source: str | int):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Не вдалося відкрити джерело відео: {source}")
    return capture


def create_video_writer(
    output_path: str | Path,
    fps: float,
    frame_width: int,
    frame_height: int,
    codec: str = "XVID",
):
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise ValueError(f"Не вдалося створити запис відео для файлу: {destination}")
    return writer


def read_source_fps(capture, fallback_fps: float = 25.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1 or fps > 240:
        return fallback_fps
    return fps


def read_frame_size(capture) -> tuple[int, int]:
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Не вдалося зчитати розмір кадру з джерела відео")

    return frame_width, frame_height
