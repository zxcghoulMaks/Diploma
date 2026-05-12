from __future__ import annotations

from collections import Counter
from itertools import count

import cv2

from .config import DetectionConfig
from .object_models import BoundingBox, DetectedObject, TrackedObject
from .text_rendering import draw_unicode_texts
from .yolo_detection import YOLOObjectDetector


# Керує всією логікою детекції, трекінгу та відмалювання об'єктів на кадрі.
class ObjectTrackingSystem:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.detector = YOLOObjectDetector(config)
        self._frame_index = 0
        self._track_id_counter = count(1)
        self._tracks: list[TrackedObject] = []
        self._match_iou_threshold = self.config.track_iou_threshold
        self._max_missed_detection_cycles = self.config.track_max_missed_cycles
        self._edge_margin_ratio = 0.06
        self._detection_blend = 0.65
        self._tracker_blend = 0.35
        self._center_distance_threshold = 1.35

    def annotate(self, frame, detection_frame=None, run_detection: bool = True):
        source_frame = frame if detection_frame is None else detection_frame
        tracked_objects = self.track(source_frame, run_detection=run_detection)
        annotated_frame = frame.copy()
        text_items: list[tuple[str, tuple[int, int]]] = []

        for tracked_object in tracked_objects:
            if not self._should_render_track(tracked_object):
                continue
            cv2.rectangle(
                annotated_frame,
                tracked_object.bounding_box.top_left(),
                tracked_object.bounding_box.bottom_right(),
                self.config.box_color,
                self.config.box_thickness,
            )

        summary = self._build_summary(tracked_objects)
        text_items.append((summary, (10, 4)))

        return draw_unicode_texts(
            annotated_frame,
            text_items,
            self.config.box_color,
            self._font_size(),
            self.config.font_path,
        )

    def track(self, frame, run_detection: bool = True) -> list[TrackedObject]:
        self._frame_index += 1
        for track in self._tracks:
            track.age_frames += 1

        self._update_trackers(frame)

        if run_detection and self._should_run_detection(frame.shape[1], frame.shape[0]):
            detections = self._run_detectors(frame)
            self._merge_detections_with_tracks(frame, detections)

        self._prune_lost_tracks()
        return list(self._tracks)

    def _should_run_detection(self, frame_width: int, frame_height: int) -> bool:
        if self._frame_index == 1 or self._frame_index % self.config.detection_interval == 0:
            return True

        for track in self._tracks:
            if track.tracker is None:
                return True
            if (
                self._is_near_frame_edge(track.bounding_box, frame_width, frame_height)
                and self._frame_index % 2 == 0
            ):
                return True

        return False

    def _run_detectors(self, frame) -> list[DetectedObject]:
        return self.detector.detect(frame)

    def _update_trackers(self, frame) -> None:
        updated_tracks: list[TrackedObject] = []
        for track in self._tracks:
            if track.tracker is None:
                updated_tracks.append(track)
                continue

            success, raw_box = track.tracker.update(frame)
            if not success:
                track.tracker = None
                track.missed_detection_cycles += 1
                updated_tracks.append(track)
                continue

            tracker_box = self._from_tracker_box(raw_box, frame.shape[1], frame.shape[0])
            track.bounding_box = self._blend_boxes(
                previous_box=track.bounding_box,
                current_box=tracker_box,
                current_weight=self._tracker_blend,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            updated_tracks.append(track)

        self._tracks = updated_tracks

    def _merge_detections_with_tracks(self, frame, detections: list[DetectedObject]) -> None:
        unmatched_track_indices = set(range(len(self._tracks)))
        unmatched_detection_indices = set(range(len(detections)))

        scored_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue

                match_score = self._calculate_match_score(track.bounding_box, detection.bounding_box)
                if match_score is not None:
                    scored_pairs.append((match_score, track_index, detection_index))

        for _, track_index, detection_index in sorted(scored_pairs, reverse=True):
            if track_index not in unmatched_track_indices or detection_index not in unmatched_detection_indices:
                continue

            detection = detections[detection_index]
            track = self._tracks[track_index]
            track.bounding_box = self._blend_boxes(
                previous_box=track.bounding_box,
                current_box=detection.bounding_box,
                current_weight=self._detection_blend,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
            track.confidence = detection.confidence
            track.display_label = detection.display_label
            track.missed_detection_cycles = 0
            track.detection_hits += 1
            track.tracker = self._create_initialized_tracker(frame, detection.bounding_box)

            unmatched_track_indices.remove(track_index)
            unmatched_detection_indices.remove(detection_index)

        for track_index in unmatched_track_indices:
            self._tracks[track_index].missed_detection_cycles += 1

        for detection_index in unmatched_detection_indices:
            detection = detections[detection_index]
            self._tracks.append(
                TrackedObject(
                    track_id=next(self._track_id_counter),
                    class_name=detection.class_name,
                    display_label=detection.display_label,
                    bounding_box=detection.bounding_box,
                    confidence=detection.confidence,
                    tracker=self._create_initialized_tracker(frame, detection.bounding_box),
                    missed_detection_cycles=0,
                    age_frames=1,
                    detection_hits=1,
                )
            )

    def _prune_lost_tracks(self) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_detection_cycles <= self._allowed_missed_cycles(track)
        ]

    def _create_initialized_tracker(self, frame, bounding_box: BoundingBox):
        tracker = self._create_tracker()
        tracker.init(frame, bounding_box.as_xywh())
        return tracker

    def _create_tracker(self):
        tracker_names = (
            "TrackerKCF_create",
            "TrackerMOSSE_create",
            "TrackerMIL_create",
            "TrackerCSRT_create",
        )
        namespaces = [cv2]
        if hasattr(cv2, "legacy"):
            namespaces.append(cv2.legacy)

        for namespace in namespaces:
            for tracker_name in tracker_names:
                if hasattr(namespace, tracker_name):
                    return getattr(namespace, tracker_name)()

        raise RuntimeError("У встановленій версії OpenCV немає доступного трекера для об'єктів")

    def _from_tracker_box(self, raw_box, frame_width: int, frame_height: int) -> BoundingBox:
        x, y, width, height = raw_box
        x = max(0, int(round(x)))
        y = max(0, int(round(y)))
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        if x + width > frame_width:
            width = frame_width - x
        if y + height > frame_height:
            height = frame_height - y

        return BoundingBox(x=x, y=y, width=max(1, width), height=max(1, height))

    def _calculate_iou(self, first_box: BoundingBox, second_box: BoundingBox) -> float:
        first_x1, first_y1, first_x2, first_y2 = first_box.as_xyxy()
        second_x1, second_y1, second_x2, second_y2 = second_box.as_xyxy()

        intersection_x1 = max(first_x1, second_x1)
        intersection_y1 = max(first_y1, second_y1)
        intersection_x2 = min(first_x2, second_x2)
        intersection_y2 = min(first_y2, second_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        first_area = first_box.width * first_box.height
        second_area = second_box.width * second_box.height
        union_area = first_area + second_area - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    def _calculate_match_score(self, tracked_box: BoundingBox, detected_box: BoundingBox) -> float | None:
        iou = self._calculate_iou(tracked_box, detected_box)
        center_distance = self._normalized_center_distance(tracked_box, detected_box)

        if iou < self._match_iou_threshold and center_distance > self._center_distance_threshold:
            return None

        return iou + max(0.0, 1.0 - center_distance) * 0.5

    def _normalized_center_distance(self, first_box: BoundingBox, second_box: BoundingBox) -> float:
        first_center_x = first_box.x + first_box.width / 2
        first_center_y = first_box.y + first_box.height / 2
        second_center_x = second_box.x + second_box.width / 2
        second_center_y = second_box.y + second_box.height / 2

        distance = ((first_center_x - second_center_x) ** 2 + (first_center_y - second_center_y) ** 2) ** 0.5
        normalizer = max(
            ((first_box.width ** 2 + first_box.height ** 2) ** 0.5),
            ((second_box.width ** 2 + second_box.height ** 2) ** 0.5),
            1.0,
        )
        return distance / normalizer

    def _blend_boxes(
        self,
        previous_box: BoundingBox,
        current_box: BoundingBox,
        current_weight: float,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        previous_weight = 1.0 - current_weight
        blended_box = BoundingBox(
            x=int(round(previous_box.x * previous_weight + current_box.x * current_weight)),
            y=int(round(previous_box.y * previous_weight + current_box.y * current_weight)),
            width=int(round(previous_box.width * previous_weight + current_box.width * current_weight)),
            height=int(round(previous_box.height * previous_weight + current_box.height * current_weight)),
        )
        return self._clamp_box(blended_box, frame_width, frame_height)

    def _clamp_box(self, bounding_box: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
        x = max(0, min(bounding_box.x, frame_width - 1))
        y = max(0, min(bounding_box.y, frame_height - 1))
        width = max(1, min(bounding_box.width, frame_width - x))
        height = max(1, min(bounding_box.height, frame_height - y))
        return BoundingBox(x=x, y=y, width=width, height=height)

    def _is_near_frame_edge(self, bounding_box: BoundingBox, frame_width: int, frame_height: int) -> bool:
        margin_x = max(8, int(frame_width * self._edge_margin_ratio))
        margin_y = max(8, int(frame_height * self._edge_margin_ratio))
        x1, y1, x2, y2 = bounding_box.as_xyxy()
        return x1 <= margin_x or y1 <= margin_y or x2 >= frame_width - margin_x or y2 >= frame_height - margin_y

    def _allowed_missed_cycles(self, track: TrackedObject) -> int:
        extra_cycles = 1 if track.age_frames >= 10 else 0
        return self._max_missed_detection_cycles + extra_cycles

    def _should_render_track(self, track: TrackedObject) -> bool:
        return track.detection_hits >= 2 or track.confidence >= 0.55

    def _build_summary(self, tracked_objects: list[TrackedObject]) -> str:
        visible_tracks = [track for track in tracked_objects if self._should_render_track(track)]
        if not visible_tracks:
            return "Відстежувані об'єкти: 0"

        counts = Counter(track.display_label for track in visible_tracks)
        details = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        return f"Відстежувані об'єкти: {len(visible_tracks)} ({details})"

    def _font_size(self) -> int:
        return max(16, int(self.config.font_scale * 28))
