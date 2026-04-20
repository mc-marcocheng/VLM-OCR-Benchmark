"""Image preprocessing utilities for Chandra OCR-2.

Adapted from the official chandra repository (chandra/model/util.py).
The model requires images aligned to a 28-pixel grid within specific pixel bounds.
"""

from __future__ import annotations

from PIL import Image

# Model image constraints
MAX_SIZE = (3072, 2048)  # max width, max height → max 6,291,456 pixels
MIN_SIZE = (1792, 28)  # min width, min height → min 50,176 pixels
GRID_SIZE = 28


def scale_to_fit(
    img: Image.Image,
    max_size: tuple[int, int] = MAX_SIZE,
    min_size: tuple[int, int] = MIN_SIZE,
    grid_size: int = GRID_SIZE,
) -> Image.Image:
    """Resize image to fit within model's pixel constraints.

    The image is scaled so that total pixels fall within [min_pixels, max_pixels],
    and both dimensions are aligned to `grid_size` (28 pixels). Aspect ratio is
    preserved as closely as possible.

    Args:
        img: Input PIL Image.
        max_size: (max_width, max_height) tuple for upper pixel bound.
        min_size: (min_width, min_height) tuple for lower pixel bound.
        grid_size: Dimensions must be divisible by this value.

    Returns:
        Resized PIL Image with grid-aligned dimensions.
    """
    resample_method = Image.Resampling.LANCZOS

    width, height = img.size

    # Check for empty or invalid image
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    # 1. Determine ideal float scale based on pixel bounds
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    # 2. Convert dimensions to integer "grid blocks"
    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    # 3. Refinement Loop: Ensure we are under the max limit
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break

        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue

        # Compare distortion: Which move preserves Aspect Ratio better?
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)

        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    # 4. Calculate final pixel dimensions
    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size

    # Return original if no changes were needed
    if (new_width, new_height) == (width, height):
        return img

    return img.resize((new_width, new_height), resample=resample_method)
