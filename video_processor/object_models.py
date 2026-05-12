from __future__ import annotations

from dataclasses import dataclass


# Описує рамку навколо знайденого об'єкта на кадрі.
@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def top_left(self) -> tuple[int, int]:
        return (self.x, self.y)

    def bottom_right(self) -> tuple[int, int]:
        return (self.x + self.width, self.y + self.height)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


# Описує один результат детекції для конкретного класу об'єкта.
@dataclass(frozen=True)
class DetectedObject:
    class_name: str
    display_label: str
    bounding_box: BoundingBox
    confidence: float


# Описує об'єкт, який уже має стабільний track id між кадрами.
@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    display_label: str
    bounding_box: BoundingBox
    confidence: float
    tracker: object | None
    missed_detection_cycles: int = 0
    age_frames: int = 0
    detection_hits: int = 1
