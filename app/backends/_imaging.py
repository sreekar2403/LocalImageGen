"""PIL helpers shared by every image-producing backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def apply_text_overlay(
    image: "Image.Image",
    text: str,
    position: str = "bottom",
    color: str = "#FFFFFF",
    bg_color: str = "#00000080",
    font_size: int = 48,
    padding: int = 20,
) -> "Image.Image":
    """Apply text overlay to an image using PIL.

    Args:
        image: The image to overlay text on
        text: The text to render
        position: Where to place text (top, bottom, center, top-left, top-right, bottom-left, bottom-right)
        color: Text color (hex)
        bg_color: Background color with optional alpha (hex)
        font_size: Font size in pixels
        padding: Padding around text

    Returns:
        Image with text overlay
    """
    from PIL import Image, ImageDraw, ImageFont

    # Make a copy to avoid modifying the original
    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default
    try:
        # Try common system fonts
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    img_width, img_height = img.size

    if position == "top":
        x = (img_width - text_width) // 2
        y = padding
    elif position == "bottom":
        x = (img_width - text_width) // 2
        y = img_height - text_height - padding
    elif position == "center":
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    elif position == "top-left":
        x = padding
        y = padding
    elif position == "top-right":
        x = img_width - text_width - padding
        y = padding
    elif position == "bottom-left":
        x = padding
        y = img_height - text_height - padding
    elif position == "bottom-right":
        x = img_width - text_width - padding
        y = img_height - text_height - padding
    else:
        # Default to bottom center
        x = (img_width - text_width) // 2
        y = img_height - text_height - padding

    # Parse colors
    def parse_color(color_str: str) -> tuple:
        """Parse hex color string to RGBA tuple."""
        color_str = color_str.lstrip("#")
        if len(color_str) == 6:
            r, g, b = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
            return (r, g, b, 255)
        elif len(color_str) == 8:
            r, g, b, a = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16), int(color_str[6:8], 16)
            return (r, g, b, a)
        return (255, 255, 255, 255)

    text_color_rgba = parse_color(color)
    bg_color_rgba = parse_color(bg_color)

    # Draw background rectangle
    bg_x1 = x - padding
    bg_y1 = y - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + text_height + padding
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color_rgba)

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color_rgba)

    return img.convert("RGB")
