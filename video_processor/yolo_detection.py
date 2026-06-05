from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig
from .object_models import BoundingBox, DetectedObject


DEFAULT_CLASS_IDS = {
    "person": 0,
    "dog": 16,
}


class YOLOObjectDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "YOLO model file was not found. "
                f"Expected file: {self.model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.enabled_class_ids = self._build_enabled_class_ids()

    def detect(self, frame) -> list[DetectedObject]:
        detections = self._detect_single_frame(frame) if self.config.detect_full_frame else []
        frame_height, frame_width = frame.shape[:2]

        for x1, y1, x2, y2 in self.config.detection_regions:
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(x1 + 1, min(x2, frame_width))
            y2 = max(y1 + 1, min(y2, frame_height))

            crop = frame[y1:y2, x1:x2]
            crop_detections = self._detect_single_frame(crop)
            detections.extend(
                self._offset_detection(detection, x1, y1, frame_width, frame_height)
                for detection in crop_detections
            )

        detections = [
            detection
            for detection in detections
            if not self._is_inside_excluded_region(detection.bounding_box)
        ]
        return self._merge_duplicate_detections(detections)

    def _is_inside_excluded_region(self, bounding_box: BoundingBox) -> bool:
        box_x1, box_y1, box_x2, box_y2 = bounding_box.as_xyxy()
        box_area = max(1, bounding_box.width * bounding_box.height)

        for x1, y1, x2, y2 in self.config.excluded_detection_regions:
            intersection_width = max(0, min(box_x2, x2) - max(box_x1, x1))
            intersection_height = max(0, min(box_y2, y2) - max(box_y1, y1))
            intersection_area = intersection_width * intersection_height
            if intersection_area / box_area >= 0.25:
                return True

        return False

    def _detect_single_frame(self, frame) -> list[DetectedObject]:
        input_height, input_width = self.config.model_input_size
        blob, scale, pad_x, pad_y = self._build_blob(frame, input_width, input_height)

        self.net.setInput(blob)
        try:
            outputs = self.net.forward()
        except cv2.error as error:
            raise RuntimeError(
                "OpenCV DNN could not run YOLO ONNX. "
                f"Current model_input_size={self.config.model_input_size} is not compatible with this model. "
                "For bundled models/yolov8n.onnx use [640, 640], or provide an ONNX model "
                "exported for the configured input size."
            ) from error
        predictions = self._reshape_predictions(outputs)

        if predictions.shape[1] in (6, 7):
            return self._parse_postprocessed_predictions(
                predictions=predictions,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                confidence_threshold=self.config.confidence_threshold,
            )

        return self._parse_raw_predictions(
            predictions=predictions,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            input_width=input_width,
            input_height=input_height,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            confidence_threshold=self.config.confidence_threshold,
        )

    def _offset_detection(
        self,
        detection: DetectedObject,
        offset_x: int,
        offset_y: int,
        frame_width: int,
        frame_height: int,
    ) -> DetectedObject:
        box = detection.bounding_box
        x = max(0, min(box.x + offset_x, frame_width - 1))
        y = max(0, min(box.y + offset_y, frame_height - 1))
        width = max(1, min(box.width, frame_width - x))
        height = max(1, min(box.height, frame_height - y))
        return DetectedObject(
            class_name=detection.class_name,
            display_label=detection.display_label,
            bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
            confidence=detection.confidence,
        )

    def _merge_duplicate_detections(self, detections: list[DetectedObject]) -> list[DetectedObject]:
        if len(detections) <= 1:
            return detections

        merged: list[DetectedObject] = []
        for class_name in sorted({detection.class_name for detection in detections}):
            class_detections = [
                detection for detection in detections if detection.class_name == class_name
            ]
            boxes = [list(detection.bounding_box.as_xywh()) for detection in class_detections]
            scores = [detection.confidence for detection in class_detections]
            kept_indices = cv2.dnn.NMSBoxes(
                bboxes=boxes,
                scores=scores,
                score_threshold=self.config.confidence_threshold,
                nms_threshold=self.config.nms_threshold,
            )
            if kept_indices is None or len(kept_indices) == 0:
                continue
            merged.extend(class_detections[index] for index in kept_indices.flatten().tolist())

        return merged

    def _parse_raw_predictions(
        self,
        predictions: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        input_width: int,
        input_height: int,
        frame_width: int,
        frame_height: int,
        confidence_threshold: float,
    ) -> list[DetectedObject]:
        candidate_boxes: list[list[int]] = []
        candidate_scores: list[float] = []
        candidate_class_names: list[str] = []
        candidate_display_labels: list[str] = []

        for prediction in predictions:
            class_name, class_score = self._extract_class_score(prediction)
            if class_name is None or class_score < confidence_threshold:
                continue

            bounding_box = self._decode_bounding_box(
                prediction=prediction,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                input_width=input_width,
                input_height=input_height,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if bounding_box.width < self.config.min_size[0] or bounding_box.height < self.config.min_size[1]:
                continue
            if not self._passes_class_specific_filters(class_name, bounding_box):
                continue

            candidate_boxes.append(list(bounding_box.as_xywh()))
            candidate_scores.append(class_score)
            candidate_class_names.append(class_name)
            candidate_display_labels.append(self.config.class_labels.get(class_name, class_name))

        if not candidate_boxes:
            return []

        kept_indices = cv2.dnn.NMSBoxes(
            bboxes=candidate_boxes,
            scores=candidate_scores,
            score_threshold=confidence_threshold,
            nms_threshold=self.config.nms_threshold,
        )
        if kept_indices is None or len(kept_indices) == 0:
            return []

        normalized_indices = kept_indices.flatten().tolist()
        return [
            DetectedObject(
                class_name=candidate_class_names[index],
                display_label=candidate_display_labels[index],
                bounding_box=BoundingBox(*candidate_boxes[index]),
                confidence=float(candidate_scores[index]),
            )
            for index in normalized_indices
        ]

    def _build_enabled_class_ids(self) -> dict[int, str]:
        enabled: dict[int, str] = {}
        for class_name in self.config.enabled_object_classes:
            if class_name not in DEFAULT_CLASS_IDS:
                supported_classes = ", ".join(sorted(DEFAULT_CLASS_IDS))
                raise ValueError(
                    f"YOLO detector has no class id for '{class_name}'. "
                    f"Supported classes: {supported_classes}"
                )
            enabled[DEFAULT_CLASS_IDS[class_name]] = class_name
        return enabled

    def _build_blob(self, frame, input_width: int, input_height: int) -> tuple[np.ndarray, float, float, float]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized_width = min(resized_width, input_width)
        resized_height = min(resized_height, input_height)

        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        letterboxed_frame = np.full((input_height, input_width, 3), 114, dtype=np.uint8)

        x_offset = (input_width - resized_width) // 2
        y_offset = (input_height - resized_height) // 2
        letterboxed_frame[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_frame

        blob = cv2.dnn.blobFromImage(
            image=letterboxed_frame,
            scalefactor=1 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
        )
        return blob, scale, float(x_offset), float(y_offset)

    def _reshape_predictions(self, outputs) -> np.ndarray:
        squeezed = np.squeeze(outputs)
        if squeezed.ndim != 2:
            raise ValueError("YOLO returned an unexpected model output shape")

        if squeezed.shape[0] in self._supported_prediction_widths():
            return squeezed.T
        if squeezed.shape[1] in self._supported_prediction_widths():
            return squeezed

        if squeezed.shape[0] > squeezed.shape[1]:
            return squeezed
        return squeezed.T

    def _supported_prediction_widths(self) -> set[int]:
        raw_prediction_widths = {4 + len(self.enabled_class_ids), 5 + len(self.enabled_class_ids)}
        return {6, 7, 84, 85, *raw_prediction_widths}

    def _extract_class_score(self, prediction: np.ndarray) -> tuple[str | None, float]:
        has_objectness = len(prediction) >= 85
        class_scores = prediction[5:] if has_objectness else prediction[4:]
        if class_scores.size == 0:
            return None, 0.0

        class_index = int(np.argmax(class_scores))
        class_name = self.enabled_class_ids.get(class_index)
        if class_name is None:
            return None, 0.0

        if has_objectness:
            score = float(prediction[4] * class_scores[class_index])
        else:
            score = float(class_scores[class_index])

        return class_name, score

    def _decode_bounding_box(
        self,
        prediction: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        input_width: int,
        input_height: int,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        center_x, center_y, width, height = prediction[:4]

        if max(abs(center_x), abs(center_y), abs(width), abs(height)) <= 2.0:
            center_x *= input_width
            center_y *= input_height
            width *= input_width
            height *= input_height

        x = (center_x - width / 2 - pad_x) / scale
        y = (center_y - height / 2 - pad_y) / scale
        width = width / scale
        height = height / scale

        x = max(0, min(int(round(x)), frame_width - 1))
        y = max(0, min(int(round(y)), frame_height - 1))
        width = max(1, min(int(round(width)), frame_width - x))
        height = max(1, min(int(round(height)), frame_height - y))

        return BoundingBox(x=x, y=y, width=width, height=height)

    def _parse_postprocessed_predictions(
        self,
        predictions: np.ndarray,
        frame_width: int,
        frame_height: int,
        confidence_threshold: float,
    ) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        for prediction in predictions:
            class_name, confidence, bounding_box = self._decode_postprocessed_prediction(
                prediction,
                frame_width,
                frame_height,
            )
            if class_name is None or confidence < confidence_threshold:
                continue
            if bounding_box.width < self.config.min_size[0] or bounding_box.height < self.config.min_size[1]:
                continue
            if not self._passes_class_specific_filters(class_name, bounding_box):
                continue

            detections.append(
                DetectedObject(
                    class_name=class_name,
                    display_label=self.config.class_labels.get(class_name, class_name),
                    bounding_box=bounding_box,
                    confidence=confidence,
                )
            )
        return detections

    def _passes_class_specific_filters(self, class_name: str, bounding_box: BoundingBox) -> bool:
        if class_name != "person":
            return True

        aspect_ratio = bounding_box.height / max(1, bounding_box.width)
        area = bounding_box.width * bounding_box.height

        return (
            aspect_ratio >= self.config.person_min_aspect_ratio
            and aspect_ratio <= self.config.person_max_aspect_ratio
            and area >= self.config.person_min_area
        )

    def _decode_postprocessed_prediction(
        self,
        prediction: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[str | None, float, BoundingBox]:
        empty_box = BoundingBox(0, 0, 1, 1)

        if len(prediction) == 6:
            x1, y1, x2, y2, confidence, class_id = prediction
        elif len(prediction) == 7:
            _, class_id, confidence, x1, y1, x2, y2 = prediction
        else:
            return None, 0.0, empty_box

        class_name = self.enabled_class_ids.get(int(class_id))
        if class_name is None:
            return None, 0.0, empty_box

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
            x1 *= frame_width
            x2 *= frame_width
            y1 *= frame_height
            y2 *= frame_height

        x1 = max(0, min(int(round(x1)), frame_width - 1))
        y1 = max(0, min(int(round(y1)), frame_height - 1))
        x2 = max(x1 + 1, min(int(round(x2)), frame_width))
        y2 = max(y1 + 1, min(int(round(y2)), frame_height))

        return (
            class_name,
            float(confidence),
            BoundingBox(
                x=x1,
                y=y1,
                width=max(1, x2 - x1),
                height=max(1, y2 - y1),
            ),
        )
