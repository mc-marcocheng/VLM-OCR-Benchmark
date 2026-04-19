"""Image utilities for Dots.MOCR."""
from __future__ import annotations

import math
from PIL import Image

from model_dots_mocr.utils.consts import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    image: Image.Image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    factor: int = IMAGE_FACTOR,
    to_gray: bool = False,
) -> Image.Image:
    """Resize image to fit within pixel constraints while maintaining aspect ratio.

    Args:
        image: PIL Image to resize.
        min_pixels: Minimum total pixels.
        max_pixels: Maximum total pixels.
        factor: Size must be divisible by this factor.
        to_gray: Convert to grayscale if True.

    Returns:
        Resized PIL Image.
    """
    if to_gray:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    width, height = image.size
    current_pixels = width * height

    # If within bounds and already aligned, return as-is
    if (min_pixels <= current_pixels <= max_pixels and
        width % factor == 0 and height % factor == 0):
        return image

    # Calculate scale factor
    if current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
    elif current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
    else:
        scale = 1.0

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Round to factor
    new_width = max(factor, round_by_factor(new_width, factor))
    new_height = max(factor, round_by_factor(new_height, factor))

    # Ensure within bounds
    while new_width * new_height > max_pixels:
        if new_width > new_height:
            new_width -= factor
        else:
            new_height -= factor

    while new_width * new_height < min_pixels:
        if new_width < new_height:
            new_width += factor
        else:
            new_height += factor

    # Ensure minimum size
    new_width = max(factor, new_width)
    new_height = max(factor, new_height)

    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image
