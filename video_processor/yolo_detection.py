from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig
from .object_models import BoundingBox, DetectedObject


COCO_CLASS_IDS = {
    "person": 0,
    "dog": 16,
}


# Детектор об'єктів на базі YOLO ONNX через OpenCV DNN.
class YOLOObjectDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Не знайдено файл моделі YOLO. "
                f"Очікується файл: {self.model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.enabled_class_ids = self._build_enabled_class_ids()

    def detect(self, frame) -> list[DetectedObject]:
        input_height, input_width = self.config.model_input_size
        blob, scale, pad_x, pad_y = self._build_blob(frame, input_width, input_height)

        self.net.setInput(blob)
        outputs = self.net.forward()
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
            if class_name not in COCO_CLASS_IDS:
                supported_classes = ", ".join(sorted(COCO_CLASS_IDS))
                raise ValueError(
                    f"YOLO-детектор не має COCO id для класу '{class_name}'. "
                    f"Підтримувані класи: {supported_classes}"
                )
            enabled[COCO_CLASS_IDS[class_name]] = class_name
        return enabled

    def _build_blob(self, frame, input_width: int, input_height: int) -> tuple[np.ndarray, float, float, float]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = int(round(frame_width * scale))
        resized_height = int(round(frame_height * scale))

        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        letterboxed_frame = np.full((input_height, input_width, 3), 114, dtype=np.uint8)

        pad_x = (input_width - resized_width) / 2
        pad_y = (input_height - resized_height) / 2
        x_offset = int(round(pad_x))
        y_offset = int(round(pad_y))
        letterboxed_frame[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_frame

        blob = cv2.dnn.blobFromImage(
            image=letterboxed_frame,
            scalefactor=1 / 255.0,
            size=(input_width, input_height),
            swapRB=True,
            crop=False,
        )
        return blob, scale, pad_x, pad_y

    def _reshape_predictions(self, outputs) -> np.ndarray:
        squeezed = np.squeeze(outputs)
        if squeezed.ndim != 2:
            raise ValueError("YOLO повернув неочікувану форму виходу моделі")

        if squeezed.shape[0] in (6, 7, 84, 85):
            return squeezed.T
        if squeezed.shape[1] in (6, 7, 84, 85):
            return squeezed

        if squeezed.shape[0] > squeezed.shape[1]:
            return squeezed
        return squeezed.T

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

        # Для людей відсікаємо занадто широкі або зовсім дрібні рамки,
        # які часто з'являються на кутах, сходах або інших контрастних деталях.
        return aspect_ratio >= 2.0 and area >= 500

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
