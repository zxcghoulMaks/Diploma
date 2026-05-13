from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import cv2

from .config import AppConfig, load_config
from .display import DisplayWindow
from .filters import apply_configured_filters
from .object_tracking import ObjectTrackingSystem
from .ui import VideoSelection, prompt_video_selection
from .video_io import create_video_writer, open_video_source, read_frame_size, read_source_fps

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "app_config.toml"
ESCAPE_KEY = 27


def run(config: AppConfig) -> None:
    capture = open_video_source(config.video.source)
    source_fps = read_source_fps(capture)
    frame_width, frame_height = read_frame_size(capture)
    tracking_system = (
        ObjectTrackingSystem(config.detection)
        if config.detection.enabled_object_classes
        else None
    )
    output_fps = config.video.fps if config.video.fps > 0 else source_fps
    writer = create_video_writer(
        output_path=config.video.output,
        fps=output_fps,
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
    display_frame_index = 0
    playback_started_at = time.perf_counter()
    source_frame_index = 0

    try:
        while True:
            if config.video.realtime_preview:
                source_frame_index += _skip_late_frames(
                    capture=capture,
                    source_fps=source_fps,
                    playback_started_at=playback_started_at,
                    source_frame_index=source_frame_index,
                    max_frame_drop=config.video.preview_max_frame_drop,
                )

            has_frame, frame = capture.read()
            if not has_frame:
                print("Video processing finished or a frame could not be read.")
                break

            source_frame_index += 1
            processed_frame = apply_configured_filters(frame, config.filters)
            if tracking_system is not None:
                display_frame_index += 1
                should_update_tracking = (
                    display_frame_index == 1
                    or display_frame_index % config.detection.processing_stride == 0
                )
                processed_frame = tracking_system.annotate(
                    processed_frame,
                    detection_frame=frame,
                    run_detection=should_update_tracking,
                )

            if not window.show(processed_frame):
                break

            writer.write(processed_frame)

            key = window.poll_key(1)
            if key == ESCAPE_KEY:
                break
            if key in toggle_keys:
                window.toggle_fullscreen()
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video processing application built on OpenCV")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Launch processing immediately without the video picker window",
    )
    parser.add_argument(
        "--source",
        help="Override the video source from the command line",
    )
    parser.add_argument(
        "--output",
        help="Override the output file path",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.source:
        config = _apply_video_selection(
            config,
            VideoSelection(
                source=Path(args.source).resolve(),
                output=_resolve_output_override(args.output, config.video.output),
            ),
        )
    elif not args.no_ui:
        selection = prompt_video_selection(config.video.source, config.video.output)
        if selection is None:
            return 0
        config = _apply_video_selection(config, selection)

    run(config)
    return 0


def _toggle_keys(character: str) -> tuple[int, int]:
    return (ord(character.lower()), ord(character.upper()))


def _skip_late_frames(
    *,
    capture,
    source_fps: float,
    playback_started_at: float,
    source_frame_index: int,
    max_frame_drop: int,
) -> int:
    if source_fps <= 0 or max_frame_drop <= 0:
        return 0

    elapsed_seconds = time.perf_counter() - playback_started_at
    expected_frame_index = int(elapsed_seconds * source_fps)
    frames_to_drop = min(max(0, expected_frame_index - source_frame_index), max_frame_drop)

    dropped_frames = 0
    for _ in range(frames_to_drop):
        if not capture.grab():
            break
        dropped_frames += 1

    return dropped_frames


def _apply_video_selection(config: AppConfig, selection: VideoSelection) -> AppConfig:
    return replace(
        config,
        video=replace(
            config.video,
            source=str(selection.source),
            output=selection.output,
        ),
    )


def _resolve_output_override(raw_output: str | None, default_output: Path) -> Path:
    if raw_output is None:
        return default_output
    return Path(raw_output).resolve()
