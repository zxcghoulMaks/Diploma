from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/calibri.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


def draw_unicode_texts(
    frame: np.ndarray,
    texts: list[tuple[str, tuple[int, int]]],
    color: tuple[int, int, int],
    font_size: int,
    font_path: str | None = None,
) -> np.ndarray:
    """Малює кілька текстових підписів із підтримкою кирилиці за один прохід."""

    base_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay)
    font = _load_font(font_size, font_path)
    stroke_width = max(1, font_size // 12)
    padding_x = max(6, font_size // 4)
    padding_y = max(4, font_size // 6)

    # Pillow expects RGB, while OpenCV stores colors in BGR.
    rgb_color = (color[2], color[1], color[0])
    for text, origin in texts:
        text_box = drawer.textbbox(origin, text, font=font, stroke_width=stroke_width)
        background_box = (
            text_box[0] - padding_x,
            text_box[1] - padding_y,
            text_box[2] + padding_x,
            text_box[3] + padding_y,
        )
        drawer.rounded_rectangle(background_box, radius=6, fill=(0, 0, 0, 170))
        drawer.text(
            origin,
            text,
            font=font,
            fill=rgb_color,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )

    composed_image = Image.alpha_composite(base_image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed_image), cv2.COLOR_RGB2BGR)


def draw_unicode_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_size: int,
    font_path: str | None = None,
) -> np.ndarray:
    """Малює текст із підтримкою кирилиці поверх кадру."""

    return draw_unicode_texts(frame, [(text, origin)], color, font_size, font_path)


def _load_font(font_size: int, font_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        configured_font = Path(font_path)
        if configured_font.exists():
            return ImageFont.truetype(str(configured_font), font_size)

    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)

    return ImageFont.load_default()
