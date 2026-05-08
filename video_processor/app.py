from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import AppConfig, load_config
from .display import DisplayWindow
from .filters import apply_configured_filters
from .video_io import create_video_writer, open_video_source, read_frame_size

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "app_config.toml"
ESCAPE_KEY = 27


def run(config: AppConfig) -> None:
    capture = open_video_source(config.video.source)
    frame_width, frame_height = read_frame_size(capture)
    writer = create_video_writer(
        output_path=config.video.output,
        fps=config.video.fps,
        frame_width=frame_width,
        frame_height=frame_height,
        codec=config.video.codec,
    )
    window = DisplayWindow(
        name=config.window.processed_window_name,
        frame_size=(frame_width, frame_height),
        normal_max_height=config.window.normal_max_height,
    )

    toggle_keys = _toggle_keys(config.window.fullscreen_toggle_key)

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                print("Відео завершено або кадр не зчитано.")
                break

            processed_frame = apply_configured_filters(frame, config.filters)
            if not window.show(processed_frame):
                break

            writer.write(processed_frame)

            key = window.poll_key(25)
            if key == ESCAPE_KEY:
                break
            if key in toggle_keys:
                window.toggle_fullscreen()
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configurable OpenCV video processor")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the runtime configuration file",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run(config)
    return 0


def _toggle_keys(character: str) -> tuple[int, int]:
    return (ord(character.lower()), ord(character.upper()))
