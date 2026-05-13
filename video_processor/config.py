from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


# Зберігає параметри джерела відео та налаштування файлу для запису результату.
@dataclass(frozen=True)
class VideoConfig:
    source: str | int
    output: Path
    fps: float
    codec: str
    realtime_preview: bool
    preview_max_frame_drop: int


# Описує, які фільтри треба застосувати до кожного кадру і з якими параметрами.
@dataclass(frozen=True)
class FilterConfig:
    use_median: bool
    use_gaussian: bool
    median_kernel: int
    gaussian_kernel: tuple[int, int]


# Містить усі налаштування для пошуку людей і відмалювання рамок на відео.
@dataclass(frozen=True)
class DetectionConfig:
    enabled_object_classes: tuple[str, ...]
    model_path: str
    model_input_size: tuple[int, int]
    processing_stride: int
    detection_interval: int
    show_summary: bool
    confidence_threshold: float
    nms_threshold: float
    min_size: tuple[int, int]
    track_iou_threshold: float
    track_max_missed_cycles: int
    box_color: tuple[int, int, int]
    box_thickness: int
    font_scale: float
    font_path: str | None
    class_labels: dict[str, str]


# Зберігає параметри вікна OpenCV, у якому користувач бачить відеопотік.
@dataclass(frozen=True)
class WindowConfig:
    processed_window_name: str
    normal_max_height: int
    fullscreen_toggle_key: str


# Об'єднує всі секції конфігурації застосунку в один зручний об'єкт.
@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig
    filters: FilterConfig
    detection: DetectionConfig
    window: WindowConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    base_dir = config_path.parent
    video = raw_config.get("video", {})
    filters = raw_config.get("filters", {})
    detection = raw_config.get("detection", {})
    window = raw_config.get("window", {})

    source = _parse_source(video.get("source", 0))
    output = _resolve_path(base_dir, video.get("output", "output/result.avi"))
    fps = float(video.get("fps", 0.0))
    codec = str(video.get("codec", "MJPG")).upper()
    realtime_preview = bool(video.get("realtime_preview", False))
    preview_max_frame_drop = _validate_non_negative_integer(
        video.get("preview_max_frame_drop", 6),
        "preview_max_frame_drop",
    )

    median_kernel = _validate_odd_integer(filters.get("median_kernel", 5), "median_kernel")
    gaussian_kernel = _validate_gaussian_kernel(filters.get("gaussian_kernel", [5, 5]))
    model_path = str(_resolve_path(base_dir, detection.get("model_path", "models/yolov8n.onnx")))
    model_input_size = _validate_pair_of_positive_integers(
        detection.get("model_input_size", [640, 640]),
        "model_input_size",
    )
    min_size = _validate_pair_of_positive_integers(
        detection.get("min_size", [8, 24]),
        "min_size",
    )
    box_color = _validate_rgb_color(detection.get("box_color", [0, 255, 0]), "box_color")
    box_thickness = _validate_positive_integer(detection.get("box_thickness", 2), "box_thickness")
    font_scale = _validate_positive_float(detection.get("font_scale", 0.7), "font_scale")
    processing_stride = _validate_positive_integer(
        detection.get("processing_stride", 2),
        "processing_stride",
    )
    detection_interval = _validate_positive_integer(
        detection.get("detection_interval", 3),
        "detection_interval",
    )
    confidence_threshold = _validate_fraction(
        detection.get("confidence_threshold", 0.30),
        "confidence_threshold",
    )
    nms_threshold = _validate_fraction(detection.get("nms_threshold", 0.3), "nms_threshold")
    track_iou_threshold = _validate_fraction(
        detection.get("track_iou_threshold", 0.2),
        "track_iou_threshold",
    )
    track_max_missed_cycles = _validate_positive_integer(
        detection.get("track_max_missed_cycles", 3),
        "track_max_missed_cycles",
    )
    enabled_object_classes = _parse_enabled_object_classes(
        detection.get("enabled_object_classes"),
        detection.get("enable_person_detection"),
    )
    class_labels = _parse_class_labels(
        detection.get("labels"),
        detection.get("label"),
    )

    fullscreen_toggle_key = str(window.get("fullscreen_toggle_key", "f"))
    if len(fullscreen_toggle_key) != 1:
        raise ValueError("fullscreen_toggle_key має містити рівно один символ")

    return AppConfig(
        video=VideoConfig(
            source=source,
            output=output,
            fps=fps,
            codec=codec,
            realtime_preview=realtime_preview,
            preview_max_frame_drop=preview_max_frame_drop,
        ),
        filters=FilterConfig(
            use_median=bool(filters.get("use_median", False)),
            use_gaussian=bool(filters.get("use_gaussian", False)),
            median_kernel=median_kernel,
            gaussian_kernel=gaussian_kernel,
        ),
        detection=DetectionConfig(
            enabled_object_classes=enabled_object_classes,
            model_path=model_path,
            model_input_size=model_input_size,
            processing_stride=processing_stride,
            detection_interval=detection_interval,
            show_summary=bool(detection.get("show_summary", True)),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_size=min_size,
            track_iou_threshold=track_iou_threshold,
            track_max_missed_cycles=track_max_missed_cycles,
            box_color=box_color,
            box_thickness=box_thickness,
            font_scale=font_scale,
            font_path=_parse_optional_string(detection.get("font_path")),
            class_labels=class_labels,
        ),
        window=WindowConfig(
            processed_window_name=str(window.get("processed_window_name", "Оброблене відео")),
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
    raise ValueError("video.source має бути цілим індексом камери або рядком зі шляхом")


def _resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _validate_odd_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise ValueError(f"{field_name} має бути додатним непарним цілим числом")
    return value


def _validate_positive_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} має бути додатним цілим числом")
    return value


def _validate_non_negative_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} РјР°С” Р±СѓС‚Рё РЅРµРІС–Рґ'С”РјРЅРёРј С†С–Р»РёРј С‡РёСЃР»РѕРј")
    return value


def _validate_positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} має бути додатним числом")
    return float(value)


def _validate_fraction(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{field_name} має бути числом у межах від 0 до 1")
    return float(value)


def _parse_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Значення шрифту має бути рядком або null")

    cleaned_value = value.strip()
    return cleaned_value or None


def _parse_enabled_object_classes(value: Any, legacy_enable_person_detection: Any) -> tuple[str, ...]:
    if value is None:
        if legacy_enable_person_detection is None:
            return ("person",)
        return ("person",) if bool(legacy_enable_person_detection) else ()

    if not isinstance(value, (list, tuple)):
        raise ValueError("enabled_object_classes має бути списком назв класів")

    parsed_classes: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Кожен елемент enabled_object_classes має бути непорожнім рядком")
        parsed_classes.append(item.strip().lower())

    return tuple(dict.fromkeys(parsed_classes))


def _parse_class_labels(raw_labels: Any, legacy_person_label: Any) -> dict[str, str]:
    labels = {
        "person": "Людина",
        "dog": "Собака",
    }

    if isinstance(legacy_person_label, str) and legacy_person_label.strip():
        labels["person"] = legacy_person_label.strip()

    if raw_labels is None:
        return labels
    if not isinstance(raw_labels, dict):
        raise ValueError("Секція detection.labels має бути таблицею TOML")

    for class_name, display_label in raw_labels.items():
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError("Ключі в detection.labels мають бути непорожніми рядками")
        if not isinstance(display_label, str) or not display_label.strip():
            raise ValueError("Значення в detection.labels мають бути непорожніми рядками")
        labels[class_name.strip().lower()] = display_label.strip()

    return labels


def _validate_pair_of_positive_integers(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} має містити рівно два значення")

    width = _validate_positive_integer(value[0], f"{field_name}[0]")
    height = _validate_positive_integer(value[1], f"{field_name}[1]")
    return (width, height)


def _validate_rgb_color(value: Any, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} має містити рівно три значення")

    channels: list[int] = []
    for index, channel in enumerate(value):
        if not isinstance(channel, int) or not 0 <= channel <= 255:
            raise ValueError(f"{field_name}[{index}] має бути цілим числом у межах від 0 до 255")
        channels.append(channel)

    return tuple(channels)


def _validate_gaussian_kernel(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("gaussian_kernel має містити рівно два значення")

    width = _validate_odd_integer(value[0], "gaussian_kernel[0]")
    height = _validate_odd_integer(value[1], "gaussian_kernel[1]")
    return (width, height)
