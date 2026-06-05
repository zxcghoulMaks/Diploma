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


@dataclass(frozen=True)
class BackgroundModelConfig:
    enabled: bool
    history: int
    var_threshold: float
    detect_shadows: bool
    learning_rate: float
    binary_threshold: int
    morph_kernel: int
    min_area: int
    draw_contours: bool
    show_mask: bool
    mask_alpha: float
    foreground_color: tuple[int, int, int]
    contour_color: tuple[int, int, int]
    contour_thickness: int


# Містить усі налаштування для пошуку людей і відмалювання рамок на відео.
@dataclass(frozen=True)
class DetectionConfig:
    enabled_object_classes: tuple[str, ...]
    model_path: str
    model_input_size: tuple[int, int]
    detect_full_frame: bool
    detection_regions: tuple[tuple[int, int, int, int], ...]
    excluded_detection_regions: tuple[tuple[int, int, int, int], ...]
    processing_stride: int
    detection_interval: int
    show_summary: bool
    confidence_threshold: float
    nms_threshold: float
    min_size: tuple[int, int]
    person_min_aspect_ratio: float
    person_max_aspect_ratio: float
    person_min_area: int
    track_iou_threshold: float
    track_max_missed_cycles: int
    use_opencv_trackers: bool
    render_min_detection_hits: int
    render_high_confidence_threshold: float
    count_direction_crossings: bool
    count_line_x: int
    count_zone: tuple[int, int, int, int]
    count_min_dx: int
    count_left_to_right_label: str
    count_right_to_left_label: str
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
    background_model: BackgroundModelConfig
    detection: DetectionConfig
    window: WindowConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    base_dir = config_path.parent
    video = raw_config.get("video", {})
    filters = raw_config.get("filters", {})
    background_model = raw_config.get("background_model", {})
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
    background_history = _validate_positive_integer(
        background_model.get("history", 500),
        "background_model.history",
    )
    background_var_threshold = _validate_positive_float(
        background_model.get("var_threshold", 16.0),
        "background_model.var_threshold",
    )
    background_learning_rate = _validate_learning_rate(
        background_model.get("learning_rate", -1.0),
        "background_model.learning_rate",
    )
    background_binary_threshold = _validate_byte_integer(
        background_model.get("binary_threshold", 200),
        "background_model.binary_threshold",
    )
    background_morph_kernel = _validate_positive_integer(
        background_model.get("morph_kernel", 3),
        "background_model.morph_kernel",
    )
    background_min_area = _validate_positive_integer(
        background_model.get("min_area", 80),
        "background_model.min_area",
    )
    background_mask_alpha = _validate_fraction(
        background_model.get("mask_alpha", 0.35),
        "background_model.mask_alpha",
    )
    background_contour_thickness = _validate_positive_integer(
        background_model.get("contour_thickness", 2),
        "background_model.contour_thickness",
    )
    background_foreground_color = _validate_rgb_color(
        background_model.get("foreground_color", [0, 128, 255]),
        "background_model.foreground_color",
    )
    background_contour_color = _validate_rgb_color(
        background_model.get("contour_color", [0, 255, 255]),
        "background_model.contour_color",
    )
    model_path = str(_resolve_path(base_dir, detection.get("model_path", "models/yolov8n.onnx")))
    model_input_size = _validate_pair_of_positive_integers(
        detection.get("model_input_size", [640, 640]),
        "model_input_size",
    )
    detect_full_frame = bool(detection.get("detect_full_frame", True))
    detection_regions = _parse_detection_regions(detection.get("detection_regions"))
    excluded_detection_regions = _parse_detection_regions(
        detection.get("excluded_detection_regions")
    )
    min_size = _validate_pair_of_positive_integers(
        detection.get("min_size", [8, 24]),
        "min_size",
    )
    person_min_aspect_ratio = _validate_positive_float(
        detection.get("person_min_aspect_ratio", 1.4),
        "person_min_aspect_ratio",
    )
    person_max_aspect_ratio = _validate_positive_float(
        detection.get("person_max_aspect_ratio", 5.0),
        "person_max_aspect_ratio",
    )
    if person_max_aspect_ratio < person_min_aspect_ratio:
        raise ValueError("person_max_aspect_ratio має бути більшим або рівним person_min_aspect_ratio")
    person_min_area = _validate_positive_integer(
        detection.get("person_min_area", 120),
        "person_min_area",
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
    use_opencv_trackers = bool(detection.get("use_opencv_trackers", True))
    render_min_detection_hits = _validate_positive_integer(
        detection.get("render_min_detection_hits", 2),
        "render_min_detection_hits",
    )
    render_high_confidence_threshold = _validate_fraction(
        detection.get("render_high_confidence_threshold", 0.75),
        "render_high_confidence_threshold",
    )
    count_direction_crossings = bool(
        detection.get("count_direction_crossings", detection.get("count_left_stair_climbers", False))
    )
    count_line_x = _validate_non_negative_integer(
        detection.get("count_line_x", detection.get("count_entry_max_x", 280)),
        "count_line_x",
    )
    count_zone = _validate_rectangle(
        detection.get("count_zone", detection.get("count_stair_zone", [0, 0, 100000, 100000])),
        "count_zone",
    )
    count_min_dx = _validate_positive_integer(
        detection.get("count_min_dx", 60),
        "count_min_dx",
    )
    count_left_to_right_label = (
        _parse_optional_string(detection.get("count_left_to_right_label"))
        or _parse_optional_string(detection.get("count_label"))
        or "Left to right"
    )
    count_right_to_left_label = (
        _parse_optional_string(detection.get("count_right_to_left_label"))
        or "Right to left"
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
        background_model=BackgroundModelConfig(
            enabled=bool(background_model.get("enabled", False)),
            history=background_history,
            var_threshold=background_var_threshold,
            detect_shadows=bool(background_model.get("detect_shadows", True)),
            learning_rate=background_learning_rate,
            binary_threshold=background_binary_threshold,
            morph_kernel=background_morph_kernel,
            min_area=background_min_area,
            draw_contours=bool(background_model.get("draw_contours", True)),
            show_mask=bool(background_model.get("show_mask", False)),
            mask_alpha=background_mask_alpha,
            foreground_color=background_foreground_color,
            contour_color=background_contour_color,
            contour_thickness=background_contour_thickness,
        ),
        detection=DetectionConfig(
            enabled_object_classes=enabled_object_classes,
            model_path=model_path,
            model_input_size=model_input_size,
            detect_full_frame=detect_full_frame,
            detection_regions=detection_regions,
            excluded_detection_regions=excluded_detection_regions,
            processing_stride=processing_stride,
            detection_interval=detection_interval,
            show_summary=bool(detection.get("show_summary", True)),
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            min_size=min_size,
            person_min_aspect_ratio=person_min_aspect_ratio,
            person_max_aspect_ratio=person_max_aspect_ratio,
            person_min_area=person_min_area,
            track_iou_threshold=track_iou_threshold,
            track_max_missed_cycles=track_max_missed_cycles,
            use_opencv_trackers=use_opencv_trackers,
            render_min_detection_hits=render_min_detection_hits,
            render_high_confidence_threshold=render_high_confidence_threshold,
            count_direction_crossings=count_direction_crossings,
            count_line_x=count_line_x,
            count_zone=count_zone,
            count_min_dx=count_min_dx,
            count_left_to_right_label=count_left_to_right_label,
            count_right_to_left_label=count_right_to_left_label,
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


def _validate_byte_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{field_name} must be an integer from 0 to 255")
    return value


def _validate_positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} має бути додатним числом")
    return float(value)


def _validate_learning_rate(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    parsed_value = float(value)
    if parsed_value == -1.0 or 0 <= parsed_value <= 1:
        return parsed_value
    raise ValueError(f"{field_name} must be -1 or a number from 0 to 1")


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


def _validate_rectangle(value: Any, field_name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} має містити рівно чотири значення")

    x1 = _validate_non_negative_integer(value[0], f"{field_name}[0]")
    y1 = _validate_non_negative_integer(value[1], f"{field_name}[1]")
    x2 = _validate_non_negative_integer(value[2], f"{field_name}[2]")
    y2 = _validate_non_negative_integer(value[3], f"{field_name}[3]")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} має бути прямокутником [x1, y1, x2, y2]")
    return (x1, y1, x2, y2)


def _parse_detection_regions(value: Any) -> tuple[tuple[int, int, int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("detection_regions має бути списком прямокутників")

    regions: list[tuple[int, int, int, int]] = []
    for index, region in enumerate(value):
        regions.append(_validate_rectangle(region, f"detection_regions[{index}]"))
    return tuple(regions)


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
