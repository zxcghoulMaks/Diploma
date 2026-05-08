from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class VideoConfig:
    source: str | int
    output: Path
    fps: float
    codec: str


@dataclass(frozen=True)
class FilterConfig:
    use_median: bool
    use_gaussian: bool
    median_kernel: int
    gaussian_kernel: tuple[int, int]


@dataclass(frozen=True)
class WindowConfig:
    processed_window_name: str
    normal_max_height: int
    fullscreen_toggle_key: str


@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig
    filters: FilterConfig
    window: WindowConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    base_dir = config_path.parent
    video = raw_config.get("video", {})
    filters = raw_config.get("filters", {})
    window = raw_config.get("window", {})

    source = _parse_source(video.get("source", 0))
    output = _resolve_path(base_dir, video.get("output", "output/result.avi"))
    fps = float(video.get("fps", 20.0))
    codec = str(video.get("codec", "XVID")).upper()

    median_kernel = _validate_odd_integer(filters.get("median_kernel", 5), "median_kernel")
    gaussian_kernel = _validate_gaussian_kernel(filters.get("gaussian_kernel", [5, 5]))

    fullscreen_toggle_key = str(window.get("fullscreen_toggle_key", "f"))
    if len(fullscreen_toggle_key) != 1:
        raise ValueError("fullscreen_toggle_key must contain exactly one character")

    return AppConfig(
        video=VideoConfig(
            source=source,
            output=output,
            fps=fps,
            codec=codec,
        ),
        filters=FilterConfig(
            use_median=bool(filters.get("use_median", True)),
            use_gaussian=bool(filters.get("use_gaussian", False)),
            median_kernel=median_kernel,
            gaussian_kernel=gaussian_kernel,
        ),
        window=WindowConfig(
            processed_window_name=str(window.get("processed_window_name", "Processed Video")),
            normal_max_height=int(window.get("normal_max_height", 700)),
            fullscreen_toggle_key=fullscreen_toggle_key,
        ),
    )


def _parse_source(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str):
        return value
    raise ValueError("video.source must be an integer camera index or a string path")


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _validate_odd_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise ValueError(f"{field_name} must be a positive odd integer")
    return value


def _validate_gaussian_kernel(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("gaussian_kernel must contain exactly two values")

    width = _validate_odd_integer(value[0], "gaussian_kernel[0]")
    height = _validate_odd_integer(value[1], "gaussian_kernel[1]")
    return (width, height)
