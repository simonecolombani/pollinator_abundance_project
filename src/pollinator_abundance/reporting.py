from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pollinator_abundance.image_processing import (
    discrete_palette,
    linear_gradient,
)
from importlib.resources import files
import pollinator_abundance

SCALE_FACTOR = 40000
SCALE_FACTOR_GRAY = 65535
PALETTE_BLACK_RED_GREEN = [(14, 10, 7), (229, 37, 31), (251, 189, 90), (41, 182, 53)]
PALETTE_INPUT = [
    (141, 9, 37),
    (255, 188, 110),
    (254, 255, 190),
    (146, 208, 103),
    (5, 94, 51),
]
PALETTE_PN = [(42, 47, 113), (148, 51, 138), (235, 113, 88), (235, 227, 51)]
# PALETTE_RED_GREEN = [(229, 37, 31), (251, 189, 90), (41, 182, 53)]
PALETTE_RED_GREEN = [
    (229, 37, 31),
    (230, 49, 35),
    (232, 62, 40),
    (234, 75, 45),
    (236, 87, 50),
    (238, 100, 55),
    (240, 113, 60),
    (241, 125, 65),
    (243, 138, 70),
    (245, 151, 75),
    (247, 163, 80),
    (249, 176, 85),
    (251, 189, 90),
    (233, 188, 86),
    (216, 187, 83),
    (198, 187, 80),
    (181, 186, 77),
    (163, 186, 74),
    (146, 185, 71),
    (128, 184, 68),
    (111, 184, 65),
    (93, 183, 62),
    (76, 183, 59),
    (58, 182, 56),
    (41, 182, 53),
]
SCALE_NUMBER_SPACE_X = 30  # Spazio per i numeri
SCALE_NUMBER_SPACE_Y = 50
TEXT_AXIS_MARGIN = 20
FONT_SIZE_BOLD = 26
FONT_SIZE_NORMAL = 20


def resize_image_reporting(image: Image.Image, scale_factor: float) -> Image.Image:
    """Resizes a PIL Image intended for reporting using a scale factor.

    Uses bilinear interpolation for resizing.

    Args:
        image: The input PIL Image object.
        scale_factor: The factor by which to scale the image dimensions.

    Returns:
        The resized PIL Image object.
    """
    # Compute new dimensions
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    # Resize (using bilinear interpolation)
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)  # type: ignore[attr-defined]
    return resized_image


def add_colorbar(
    image: Image.Image,
    palette: List[Tuple[int, int, int]],
    min_value: Union[float, int, str],
    max_value: Union[float, int, str],
    title_bar: str,
    original_height: int,  # Note: original_height seems unused, height taken from image.size
    font_regular: ImageFont.FreeTypeFont,
    font_semibold: ImageFont.FreeTypeFont,
    cbar_digits: int = 1,
    middle_cbar_tick: Union[bool, str] = True,
) -> Image.Image:
    """Adds a vertical color bar with ticks and title to the right side of an image.

    Draws a color gradient based on the provided palette, adds ticks and labels
    for min, max, and optionally middle values, and places a vertically rotated
    title next to the color bar.

    Args:
        image: The input PIL Image object to which the color bar will be added.
        palette: A list of RGB tuples defining the colors for the gradient.
        min_value: The minimum value for the color scale. Can be numeric or string.
        max_value: The maximum value for the color scale. Can be numeric or string.
        title_bar: The text title for the color bar (will be rotated 90 degrees).
        original_height: (Deprecated/Unused) The original height of the image.
                         Height is now taken directly from the input `image`.
        font_regular: PIL ImageFont object for tick labels.
        font_semibold: PIL ImageFont object for the color bar title.
        cbar_digits: The number of decimal places to display for numeric tick labels.
                     Defaults to 1.
        middle_cbar_tick: If True, adds a tick/label at the midpoint. If a string,
                          uses the string as the label for the midpoint tick. If False,
                          no middle tick is added. Defaults to True.

    Returns:
        A new PIL Image object containing the original image and the added color bar.
    """
    # Constants
    colorbar_width = 50
    tick_length = 6
    label_padding = 5  # Space between colorbar and labels
    margin_left = 1  # Left margin for the colorbar
    title_padding = 5  # Space between colorbar and title

    image_width, image_height = image.size
    total_colorbar_height = image_height
    colorbar_padding_height = 40
    colorbar_height = total_colorbar_height - 2 * colorbar_padding_height

    # Create a temporary ImageDraw object to calculate text sizes
    temp_img = Image.new("RGB", (1, 1))  # noqa: F841

    # Prepare tick labels
    labels = []
    if not isinstance(min_value, str):
        labels.append(str(round(min_value, cbar_digits)))
    else:
        labels.append(min_value)
    if not isinstance(max_value, str):
        labels.append(str(round(max_value, cbar_digits)))
    else:
        labels.append(max_value)
    positions = [total_colorbar_height - 2 * colorbar_padding_height, 3]
    if middle_cbar_tick is True:
        labels.append(str(round((min_value + max_value) / 2, cbar_digits)))  # type: ignore[operator]
        positions.append(image_height / 2 - colorbar_padding_height)  # type: ignore[arg-type]
    elif isinstance(middle_cbar_tick, str):
        labels.append(middle_cbar_tick)
        positions.append(image_height / 2 - colorbar_padding_height)  # type: ignore[arg-type]

    # Calculate maximum label width
    label_widths = [text_size(font_regular, text)[0] for text in labels]
    max_label_width = max(label_widths) if label_widths else 0

    # Calculate the total colorbar width including labels
    total_colorbar_width = (
        colorbar_width
        + tick_length
        + label_padding
        + max_label_width
        + title_padding
        + 20
    )  # Extra 20 for safety

    # Create the colorbar image
    colorbar = Image.new("RGBA", (total_colorbar_width, total_colorbar_height), "white")
    draw = ImageDraw.Draw(colorbar)

    # Draw the color gradient
    step_height = colorbar_height / len(palette)
    step_width = 30
    for i, color in enumerate(palette):
        y0 = int(round(i * step_height))
        y1 = int(round((i + 1) * step_height))
        if y1 > image_height:
            y1 = image_height
        if y1 > y0:
            draw.rectangle([0, y0, step_width, y1], fill=tuple(color))

    # Draw ticks and labels
    for label, position in zip(labels, positions):
        # Draw tick
        draw.line(
            [(step_width, position), (step_width + tick_length, position)],
            fill="black",
            width=1,
        )
        # Draw label
        text_width, text_height = text_size(font_regular, label)
        draw.text(
            (step_width + tick_length + label_padding, position - text_height / 2),
            label,
            fill="black",
            font=font_regular,
        )

    # Prepare and paste the title
    title_width, title_height = text_size(font_semibold, title_bar)
    title_image = Image.new(
        "RGBA", (title_width + 10, title_height + 10), (255, 255, 255, 0)
    )
    title_draw = ImageDraw.Draw(title_image)
    title_draw.text((0, 0), title_bar, fill="black", font=font_semibold)
    rotated_title = title_image.rotate(90, expand=1)

    # Calculate position for the rotated title
    title_x = step_width + tick_length + label_padding + max_label_width + title_padding
    title_y = (image_height - rotated_title.height) // 2

    # Paste the rotated title onto the colorbar
    colorbar.paste(rotated_title, (title_x, title_y), rotated_title)

    # Combine the original image and the colorbar
    combined_width = image_width + total_colorbar_width + margin_left
    combined_height = (
        image_height  # Ensures the combined image height matches the original image
    )
    combined = Image.new("RGBA", (combined_width, combined_height), "white")

    # Paste the original image
    combined.paste(image.convert("RGBA"), (0, 0))

    # Paste the colorbar
    colorbar_x = image_width + margin_left
    combined.paste(colorbar, (colorbar_x, 0), 0)  # type: ignore[arg-type]

    return combined


def text_size(font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int]:
    """Calculates the bounding box width and height for a given text and font.

    Args:
        font: The PIL ImageFont object to use.
        text: The string for which to calculate dimensions.

    Returns:
        A tuple containing (width, height) of the text's bounding box in pixels.
    """
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height  # type: ignore[return-value]


def add_scale_to_image(
    image: Image.Image,
    title_x: str,
    title_y: str,
    minx: Union[float, int],
    miny: Union[float, int],
    maxx: Union[float, int],
    maxy: Union[float, int],
    font_regular: ImageFont.FreeTypeFont,
    font_semibold: ImageFont.FreeTypeFont,
    font_regular_bytes: bytes,
    ticks: int = 5,
) -> Image.Image:
    """Adds X and Y axes with scales, ticks, and titles around an image.

    Creates a new, larger image with a white background, pastes the original
    image onto it with an offset, and draws X and Y axes below and to the left,
    respectively. Adds tick marks and labels based on the provided min/max values.
    Adjusts label frequency and font size for small images.

    Args:
        image: The input PIL Image object.
        title_x: The title text for the X-axis.
        title_y: The title text for the Y-axis (will be rotated 90 degrees).
        minx: The minimum value for the X-axis scale.
        miny: The minimum value for the Y-axis scale.
        maxx: The maximum value for the X-axis scale.
        maxy: The maximum value for the Y-axis scale.
        font_regular: PIL ImageFont object for tick labels.
        font_semibold: PIL ImageFont object for axis titles.
        font_regular_bytes: Raw bytes of the regular font file (used for resizing).
        ticks: The desired number of ticks (including endpoints) on each axis.
               Defaults to 5.

    Returns:
        A new PIL Image object containing the original image surrounded by axes.
    """
    tick_length = 10
    title_space = 10  # Distance between x-axis title and tick labels

    # Compute x and y axes' title text dimensions
    title_x_text_width, title_x_text_height = text_size(font_semibold, title_x)
    title_y_text_width, title_y_text_height = text_size(font_semibold, title_y)
    max_value_text = (
        f"{max(maxx, maxy):.2f}"  # Take the maximum value for text dimension
    )
    number_text_width, number_text_height = text_size(font_regular, max_value_text)

    image_offset_x = (
        number_text_width + title_space * 2 + tick_length + title_y_text_height
    )
    image_offset_y = 6

    # Calcola la nuova dimensione dell'immagine
    new_width = image.width + image_offset_x + image_offset_y * 5
    new_height = (
        image.height
        + image_offset_y * 3
        + number_text_height
        + title_space * 2
        + title_x_text_height
        + tick_length
    )
    new_image = Image.new("RGB", (new_width, new_height), "white")
    draw = ImageDraw.Draw(new_image)

    new_image.paste(image, (image_offset_x, image_offset_y))

    # Calcola gli intervalli per le tacche
    x_interval = image.width / (ticks - 1)
    y_interval = image.height / (ticks - 1)

    # Skip some labels and reduce label font if necessary (for very small images)
    label_x_skip = 1
    font_regular_x = font_regular
    if image.width < 300:
        label_x_skip = 2
        font_regular_x = ImageFont.truetype(
            BytesIO(font_regular_bytes), font_regular.size - 2
        )
    label_y_skip = 1
    font_regular_y = font_regular
    if image.height < 300:
        label_y_skip = 2
        font_regular_y = ImageFont.truetype(
            BytesIO(font_regular_bytes), font_regular.size - 2
        )

    # Aggiungi tacche e numeri per l'asse X
    for i in range(ticks):
        x = i * x_interval + image_offset_x
        draw.line(
            [
                (x, image.height + image_offset_y),
                (x, image.height + image_offset_y + tick_length),
            ],
            fill="black",
            width=1,
        )
        # Skip labels if needed
        if i % label_x_skip != 0:
            continue
        value = minx + (maxx - minx) * i / (ticks - 1)
        value_text_width, value_text_height = text_size(font_regular_x, f"{value:.2f}")
        draw.text(
            (x - value_text_width / 2, image.height + image_offset_y + tick_length + 5),
            f"{value:.2f}",
            fill="black",
            font=font_regular_x,
        )

    # Aggiungi tacche e numeri per l'asse Y
    for i in range(ticks):
        y = i * y_interval + image_offset_y
        draw.line(
            [(image_offset_x - tick_length, y), (image_offset_x, y)],
            fill="black",
            width=1,
        )
        # Skip labels if needed
        if i % label_y_skip != 0:
            continue
        value = miny + (maxy - miny) * (ticks - 1 - i) / (ticks - 1)
        value_text_height, _ = text_size(font_regular_y, f"{value:.2f}")
        draw.text(
            (
                image_offset_x - number_text_width - title_space - 5,
                y - value_text_height / 4,
            ),
            f"{value:.2f}",
            fill="black",
            font=font_regular_y,
        )

    # Aggiungi il titolo per l'asse X
    draw.text(
        ((new_width - title_x_text_width) / 2, new_height - title_x_text_height - 5),
        title_x,
        fill="black",
        font=font_semibold,
    )

    # Crea e posiziona il titolo per l'asse Y
    title_y_image = Image.new(
        "RGB", (title_y_text_width, title_y_text_height + title_space), "white"
    )
    title_y_draw = ImageDraw.Draw(title_y_image)
    title_y_draw.text((0, 0), title_y, fill="black", font=font_semibold)
    title_y_image = title_y_image.rotate(90, expand=1)
    new_image.paste(title_y_image, (5, (new_height - title_y_text_width) // 2))
    return new_image


def add_image_title(
    image: Image.Image, title: str, font_semibold: ImageFont.FreeTypeFont
) -> Image.Image:
    """Adds a centered title above an image.

    Creates a new image taller than the original, adds the title text at the top center,
    and pastes the original image below the title.

    Args:
        image: The input PIL Image object.
        title: The text for the title.
        font_semibold: The PIL ImageFont object to use for the title.

    Returns:
        A new PIL Image object with the title added above the original image.
    """
    font_size = FONT_SIZE_BOLD  # noqa: F841
    title_width, title_height = text_size(font_semibold, title)
    new_width = image.width
    new_height = image.height + title_height + 6
    new_image = Image.new("RGB", (new_width, new_height), "white")
    draw = ImageDraw.Draw(new_image)
    draw.text(
        (new_width / 2 - title_width / 2, 0), title, fill="black", font=font_semibold
    )
    new_image.paste(image, (0, title_height + 6))
    return new_image


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Crops an image to a specified bounding box and resizes it back to the original size.

    Args:
        image: The input PIL Image object.
        bbox: A tuple (left, upper, right, lower) defining the crop area.

    Returns:
        A PIL Image object containing the cropped region, resized (using bilinear
        interpolation) to the dimensions of the original input image.
    """
    cropped = image.crop(bbox)
    original_size = image.size
    resized_image = cropped.resize(original_size, Image.BILINEAR)  # type: ignore[attr-defined]
    return resized_image


def adjust_polygon(
    polygon: Union[List[Tuple[float, float]], np.ndarray],
    old_image_size: Tuple[int, int],
    new_bbox: Tuple[int, int, int, int],
) -> List[Tuple[float, float]]:
    """Adjusts polygon coordinates to match a cropped and resized image.

    This is used when an image has been cropped using `new_bbox` and then
    resized back to `old_image_size`. It translates and scales the polygon
    vertices defined relative to the original full image to be correct relative
    to the new cropped-then-resized image.

    Args:
        polygon: A list of (x, y) tuples or a NumPy array of coordinates
                 relative to the original, uncropped image.
        old_image_size: A tuple (width, height) of the original image size (which
                        is also the size of the final cropped+resized image).
        new_bbox: A tuple (left, upper, right, lower) representing the bounding
                  box used to crop the original image.

    Returns:
        A list of adjusted (x, y) tuples representing the polygon coordinates
        relative to the cropped-then-resized image.
    """
    old_width, old_height = old_image_size
    new_left, new_upper, new_right, new_lower = new_bbox
    new_width = new_right - new_left
    new_height = new_lower - new_upper

    scale_x = old_width / new_width
    scale_y = old_height / new_height

    adjusted_polygon = []
    for x, y in polygon:
        # Translate points in the new Bounding Box
        x_translated = x - new_left
        y_translated = y - new_upper

        # Scale points according to expansion factor
        x_scaled = x_translated * scale_x
        y_scaled = y_translated * scale_y

        adjusted_polygon.append((x_scaled, y_scaled))

    return adjusted_polygon


def create_image_for_reporting(
    image: Image.Image,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    title_bar: str,
    x_axis_scale: Tuple[Union[float, int], Union[float, int]],
    y_axis_scale: Tuple[Union[float, int], Union[float, int]],
    palette_scale: Optional[Tuple[Union[float, int, str], Union[float, int, str]]],
    palette: Optional[List[Tuple[int, int, int]]],
    site_pixel_polygon: Optional[Union[np.ndarray, List[np.ndarray]]],
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    value_text: Optional[str] = None,
    cbar_digits: int = 1,
    middle_cbar_tick: Union[bool, str] = True,
) -> Image.Image:
    """Creates a complete image ready for reporting with titles, scales, and optional elements.

    This function orchestrates several steps:
    1. Resizes the input image if it's too small or too large for reporting standards.
    2. Optionally crops the image based on `bounding_box`.
    3. Optionally draws site polygons (adjusting coordinates if cropped).
    4. Optionally adds a text box with `value_text`.
    5. Adds X and Y axes with scales and titles using `add_scale_to_image`.
    6. Optionally adds a color bar using `add_colorbar` if `palette` is provided.
    7. Adds a main title above the image using `add_image_title`.
    Loads required fonts from S3.

    Args:
        image: The base PIL Image object.
        title: The main title for the report image.
        x_axis_title: Title for the X-axis.
        y_axis_title: Title for the Y-axis.
        title_bar: Title for the color bar (if applicable).
        x_axis_scale: Tuple (min_x, max_x) for the X-axis scale.
        y_axis_scale: Tuple (min_y, max_y) for the Y-axis scale.
        palette_scale: Optional tuple (min_val, max_val) for the color bar scale.
                       Required if `palette` is provided. Can contain strings.
        palette: Optional list of RGB tuples defining the color palette for the color bar.
        site_pixel_polygon: Optional NumPy array (single polygon) or list of NumPy
                            arrays (multi-polygon) defining site boundaries to draw.
                            Coordinates are relative to the original `image`.
        bounding_box: Optional tuple (left, upper, right, lower) to crop the image.
                      If provided, `site_pixel_polygon` coordinates are adjusted.
                      Defaults to None (no cropping).
        value_text: Optional string to display in a white box on the image.
                    Defaults to None.
        cbar_digits: Number of decimal places for color bar labels. Defaults to 1.
        middle_cbar_tick: Setting for the middle tick on the color bar (True,
                          False, or a string label). Defaults to True.

    Returns:
        A complete PIL Image object ready for inclusion in a report.
    """

    # Load fonts data
    with open(
        files(pollinator_abundance) / "data/font_regular_bytes.txt", "rb"
    ) as binary_file_reader:  # type: ignore[call-overload]
        font_regular_bytes = binary_file_reader.read()
    font_regular = ImageFont.truetype(
        BytesIO(font_regular_bytes), size=FONT_SIZE_NORMAL
    )

    with open(
        files(pollinator_abundance) / "data/font_semibold_bytes.txt", "rb"
    ) as binary_file_reader:  # type: ignore[call-overload]
        font_semibold_bytes = binary_file_reader.read()
    font_semibold = ImageFont.truetype(
        BytesIO(font_semibold_bytes), size=FONT_SIZE_NORMAL
    )

    # Duplicate input image
    new_image = image.copy()

    # If original image is too small, increase its dimension.
    _, original_height = new_image.size
    if original_height < 300:
        new_image = resize_image_reporting(new_image, 300.0 / original_height)

    # If Bounding Box is given
    if bounding_box:
        # Crop image to Bounding Box
        new_image = crop_image(new_image, bounding_box)
        if site_pixel_polygon is not None:
            # Draw cropped image
            draw = ImageDraw.Draw(new_image, "RGBA")
            # If 'site_pixel_polygon' is a single polygon
            if isinstance(site_pixel_polygon, np.ndarray):
                # Adjust polygon coordinates
                site_pixel_polygon_adjust = adjust_polygon(
                    site_pixel_polygon, new_image.size, bounding_box
                )
                # Draw polygon
                draw.polygon(
                    [tuple(p) for p in site_pixel_polygon_adjust],
                    outline="black",
                    width=3,
                )
            # If 'site_pixel_polygon' is a multi-polygon
            else:
                # Adjust polygons' coordinates
                site_pixel_polygon_adjust = [
                    adjust_polygon(polygon, new_image.size, bounding_box)  # type: ignore[misc]
                    for polygon in site_pixel_polygon
                ]
                # Draw each polygon
                for polygon in site_pixel_polygon_adjust:
                    draw.polygon([tuple(p) for p in polygon], outline="black", width=3)  # type: ignore[arg-type]
    # Otherwise
    else:
        if site_pixel_polygon is not None:
            # Draw image
            draw = ImageDraw.Draw(new_image)
            # If 'site_pixel_polygon' is a single polygon
            if isinstance(site_pixel_polygon, np.ndarray):
                # Draw polygon
                draw.polygon(
                    [tuple(p) for p in site_pixel_polygon], outline="black", width=3
                )
            # If 'site_pixel_polygon' is a multi-polygon
            else:
                # Draw each polygon
                for polygon in site_pixel_polygon:  # type: ignore[assignment]
                    draw.polygon([tuple(p) for p in polygon], outline="black", width=3)  # type: ignore[arg-type]

    # Report image height should not exceed 600. If so, resize the image.
    _, original_height = new_image.size
    if original_height > 600:
        new_image = resize_image_reporting(new_image, 600 / original_height)
    _, original_height = new_image.size

    # Draw image
    draw = ImageDraw.Draw(new_image)
    # If a text has been given
    if value_text:
        # Set text container and dimensions
        bbox = font_semibold.getbbox(value_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        box_width = text_width + 20
        box_height = text_height + 20
        draw.rectangle([50, 50, 50 + box_width, 50 + box_height], fill="white")
        text_x = 60
        text_y = 60
        # Write text on the image
        draw.text((text_x, text_y), value_text, fill="black", font=font_semibold)

    # Add x and y axes to image
    new_image = add_scale_to_image(
        new_image,
        x_axis_title,
        y_axis_title,
        x_axis_scale[0],
        y_axis_scale[0],
        x_axis_scale[1],
        y_axis_scale[1],
        font_regular,
        font_semibold,
        font_regular_bytes,
    )

    # If a palette has been given, add a colorbar to the image
    if palette:
        new_image = add_colorbar(
            new_image,
            palette,
            palette_scale[0],  # type: ignore[index]
            palette_scale[1],  # type: ignore[index]
            title_bar,
            original_height,
            font_regular,
            font_semibold,
            cbar_digits=cbar_digits,
            middle_cbar_tick=middle_cbar_tick,
        )

    # Adjust the font size for the title to fit within the image width
    image_width, _ = new_image.size
    title_font_size = FONT_SIZE_NORMAL
    font_semibold_title = ImageFont.truetype(
        BytesIO(font_semibold_bytes), size=title_font_size
    )
    min_title_font_size = 10
    while (
        font_semibold_title.getbbox(title)[2] > image_width - 40
        and title_font_size > min_title_font_size
    ):
        title_font_size -= 1
        font_semibold_title = ImageFont.truetype(
            BytesIO(font_semibold_bytes), size=title_font_size
        )

    # Add a title to the image
    new_image = add_image_title(new_image, title, font_semibold_title)

    return new_image


def add_percentage_hectar_and_order(
    selected_keys: List[str], clc_values: List[Dict[str, Any]]
) -> Tuple[List[str], np.ndarray]:
    """Calculates percentage hectares, adds '%' column, filters, and sorts CLC data.

    Filters out entries with name "NODATA". Calculates the percentage of total
    hectares for each remaining entry. Filters out entries where the percentage
    is negligible (< 0.001%). Appends the percentage as a new column to the data
    selected by `selected_keys`. Sorts the resulting data array in descending
    order based on the 'hectare' column.

    Args:
        selected_keys: A list of keys to select from each dictionary in `clc_values`.
                       Must include 'hectare' and 'name'.
        clc_values: A list of dictionaries, where each dictionary represents a CLC
                    category and contains at least 'name' and 'hectare' keys.

    Returns:
        A tuple containing:
            - selected_keys_updated: The input `selected_keys` list with '%' appended.
            - data_sorted: A NumPy array containing the selected, filtered, augmented
                           (with percentage), and sorted data.
    """
    total_hectares = sum(d["hectare"] for d in clc_values if d["name"] != "NODATA")
    data_with_percentage = []
    for d in clc_values:
        if d["name"] != "NODATA":
            row = [d.get(key) for key in selected_keys]
            hectare_percentage = (d["hectare"] / total_hectares) * 100
            if hectare_percentage > 0.001:
                row.append(hectare_percentage)
                data_with_percentage.append(row)
    selected_keys_updated = selected_keys + ["%"]
    data_updated = np.array(data_with_percentage, dtype=object)
    hectare_index = selected_keys.index("hectare")
    data_sorted = data_updated[
        data_updated[:, hectare_index].astype(float).argsort()[::-1]
    ]
    return selected_keys_updated, data_sorted


def calculate_weighted_ns(clc_values: List[Dict[str, Any]]) -> None:
    """Calculates and adds a weighted 'ns' value to each dictionary in clc_values.

    Computes a weighted average of specific 'ns_*' columns (defined internally)
    for each dictionary in the input list. Adds the result under the key 'ns'
    to each dictionary (modifies the list in-place).

    Args:
        clc_values: A list of dictionaries representing CLC categories, expected
                    to contain keys like 'ns_soilexcavators', etc.
    """
    NS_COLUMNS = [
        "ns_soilexcavators",
        "ns_sandexcavators",
        "ns_underground_cavities",
        "ns_aboveground_cavities_wetland",
        "ns_aboveground_cavities_vegetated",
        "ns_coastal_area",
    ]
    WEIGHTS = [3, 2, 1, 1, 5, 1]
    for d in clc_values:
        ns_values = [d.get(column, 0) for column in NS_COLUMNS]
        weighted_ns = np.average(ns_values, weights=WEIGHTS)
        d["ns"] = weighted_ns


clc_csv_headers_keys_i18n = {
    "de": [
        "fa",
        "farbe",
        "hektar",
        "ns",
        "ns_bodenbagger",
        "ns_sandbagger",
        "ns_unterirdische_hohlräume",
        "ns_feuchtgebiete",
        "ns_pflanzen",
        "ns_küstengebiete",
        "ns_künstliche",
        "pn_medium",
        "msa",
        "msa_lu_tiere",
        "msa_lu_anlage",
        "msa_cc",
        "name",
    ],
    "en": [
        "fa",
        "color",
        "hectares",
        "ns",
        "ns_soilexcavators",
        "ns_sandexcavators",
        "ns_underground_cavities",
        "ns_aboveground_cavities_wetland",
        "ns_aboveground_cavities_vegetated",
        "ns_coastal_area",
        "ns_artificial",
        "pn_mean",
        "msa",
        "msa_lu_animals",
        "msa_lu_plants",
        "msa_cc",
        "name",
    ],
    "es": [
        "fa",
        "color",
        "hectáreas",
        "ns",
        "ns_excavadoras_terra",
        "ns_excavadoras_arena",
        "ns_cavidadaes_subterraneas",
        "ns_humedales",
        "ns_plantas",
        "ns_zonas_costeras",
        "ns_artificial",
        "pn_medio",
        "msa",
        "msa_lu_animales",
        "msa_lu_planta",
        "msa_cc",
        "nombre",
    ],
    "fr": [
        "fa",
        "couleur",
        "hectares",
        "ns",
        "ns_excavateurs_sol",
        "ns_excavateurs_sable",
        "ns_cavités_souterraines",
        "ns_zones_humides",
        "ns_plantes",
        "ns_zones_côtières",
        "ns_artificial",
        "pn_moyen",
        "msa",
        "msa_lu_animaux",
        "msa_lu_plantes",
        "msa_cc",
        "nom",
    ],
    "it": [
        "fa",
        "colore",
        "ettari",
        "ns",
        "ns_scavatori_suolo",
        "ns_scavatori_sabbia",
        "ns_cavità_sottosuolo",
        "ns_zone_umide",
        "ns_piante",
        "ns_aree_costiere",
        "ns_artificiali",
        "pn_medio",
        "msa",
        "msa_lu_animali",
        "msa_lu_piante",
        "msa_cc",
        "nome",
    ],
}

clc_png_headers_keys_i18n = {
    "de": ["farbe", "name", "msa", "ns", "fa", "pn_medium", "hektar", "%"],
    "en": ["color", "name", "msa", "ns", "fa", "pn_mean", "hectares", "%"],
    "es": ["color", "nombre", "msa", "ns", "fa", "pn_medio", "hectáreas", "%"],
    "fr": ["couleur", "nom", "msa", "ns", "fa", "pn_moyen", "hectares", "%"],
    "it": ["colore", "nome", "msa", "ns", "fa", "pn_medio", "ettari", "%"],
}


def get_scale_palette_units_for_layer_type(
    layer_type: str,
    values_range: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
) -> Tuple[
    Optional[Union[List, Tuple]],
    Optional[List[Tuple[int, int, int]]],
    Union[bool, str],
    Optional[str],
    int,
]:
    """Provides predefined scale, palette, units, and formatting for different layer types.

    Returns appropriate visualization parameters (color scale range, color palette,
    colorbar middle tick setting, unit string, number of digits) based on the input
    `layer_type` string. Some layer types require `values_range` to be provided.

    Args:
        layer_type: A string identifying the type of layer (e.g., "np", "fa", "msa",
                    "lst", "ndvi", "aridity").
        values_range: Optional tuple (min_val, max_val) required for layer types
                      where the scale is data-dependent (e.g., "lst", "pd", "busf").

    Returns:
        A tuple containing:
            - palette_scale: The min/max range for the color bar (List/Tuple or None).
                             Can contain numbers or strings (for discrete scales).
            - palette: A list of RGB tuples for the color bar (List or None).
            - middle_cbar_tick: Setting for the middle tick (bool or str).
            - units: A string representing the units (e.g., "[°C]", "[kg/ha/y]") or None.
            - n_digits: Suggested number of decimal places for numeric labels.

    Raises:
        ValueError: If the provided `layer_type` is not supported.
    """
    middle_cbar_tick = True
    units = None
    min_msa_cc = 0.75
    min_msa_i = 0.666666
    min_msa_f = 0.8
    n_digits = 1
    if layer_type in ["clc", "protected_areas"]:
        palette_scale = None
        palette = None
        middle_cbar_tick = False
    elif layer_type == "np":
        palette_scale = [0, 250]
        palette = linear_gradient(PALETTE_PN, n=256)[::-1]
        units = "[kg/ha/y]"
    elif layer_type == "fa":
        palette_scale = [0, 1]
        palette = linear_gradient(PALETTE_INPUT, n=256)[::-1]
    elif layer_type == "msa":
        palette_scale = [0, 1]
        palette = linear_gradient(PALETTE_BLACK_RED_GREEN, n=256)[::-1]
    elif layer_type == "msa_cc":
        palette_scale = [min_msa_cc, 1]  # type: ignore[list-item]
        palette = linear_gradient(PALETTE_RED_GREEN, n=256)[::-1]
        n_digits = 2
    elif layer_type == "msa_lu_animals" or layer_type == "msa_lu_plants":
        palette_scale = [0, 1]
        palette = linear_gradient(PALETTE_BLACK_RED_GREEN, n=256)[::-1]
    elif layer_type == "msa_i":
        palette_scale = [min_msa_i, 1]  # type: ignore[list-item]
        palette = linear_gradient(PALETTE_RED_GREEN, n=256)[::-1]
    elif layer_type == "msa_f":
        palette_scale = [min_msa_f, 1]  # type: ignore[list-item]
        palette = linear_gradient(PALETTE_RED_GREEN, n=256)[::-1]
    elif layer_type == "pa":
        palette_scale = [0, 0.4]  # type: ignore[list-item]
        palette = linear_gradient(PALETTE_INPUT, n=256)[::-1]
    elif layer_type == "ns":
        palette_scale = [0, 0.5]  # type: ignore[list-item]
        palette = linear_gradient(PALETTE_INPUT, n=256)[::-1]
    elif layer_type == "lst":
        colors = [
            "#000091",
            "#0015FF",
            "#00CCFF",
            "#4CFFAA",
            "#83FF72",
            "#F4F802",
            "#FFA300",
            "#E30000",
            "#840001",
        ]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
        units = "[°C]"
    elif layer_type == "light_pollution":
        colors = [
            "#09123B",
            "#F9E6A0",
        ]  # Dark blue (hex: #09123B) to Yellow (hex: #F9E6A0)
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
        units = "[nW/sr/cm²]"
    elif layer_type == "aridity":
        colors = [
            "#CF7563",
            "#E09053",
            "#F3BA41",
            "#FAE038",
            "#D2FA32",
            "#5DE833",
            "#40D168",
            "#49B09C",
            "#458AA1",
            "#3D5894",
        ]
        palette = discrete_palette(colors, n=10)[::-1]
        palette_scale = ["Hyper Arid", "Humid"]  # type: ignore[list-item]
        middle_cbar_tick = "Dry sub-humid"  # type: ignore[assignment]
    elif layer_type == "impermeability":
        colors = ["#88C7DF", "FEF4B1", "#FFA688"]
        palette = discrete_palette(colors, n=3)[::-1]
        palette_scale = ["Draining", "Impermeable"]  # type: ignore[list-item]
        middle_cbar_tick = "Semi-Impermeable"  # type: ignore[assignment]
    elif layer_type == "naa":
        colors = [
            "#2F4F4F",
            "#A9A9A9",
            "#7CFC00",
            "#8B4513",
            "#228B22",
            "#ADD8E6",
            "#00008B",
        ]
        palette = discrete_palette(colors, n=7)[::-1]
        palette_scale = ["Roads", "Salt water"]  # type: ignore[list-item]
        middle_cbar_tick = "Shelters"  # type: ignore[assignment]
    elif layer_type == "flood_risk":
        colors = ["#7F9127", "#147B20", "#F7A333", "#F85621", "#FB1C19"]
        palette = discrete_palette(colors, n=5)[::-1]
        palette_scale = ["Very Low", "Very High"]  # type: ignore[list-item]
        middle_cbar_tick = "Moderate"  # type: ignore[assignment]
    elif layer_type == "busf":
        colors = ["#31D136", "#D13131"]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
        units = "[%]"
    elif layer_type == "pd":
        colors = ["#F6DEDE", "#4E0202"]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
        units = "[#/km²]"
    elif layer_type == "night_lst":
        colors = [
            "#000091",
            "#0015FF",
            "#00CCFF",
            "#4CFFAA",
            "#83FF72",
            "#F4F802",
            "#FFA300",
            "#E30000",
            "#840001",
        ]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
        units = "[°C]"
    elif layer_type == "utfvi":
        colors = ["#00A67A", "#00DF80", "#FFD21E", "#FF8B16", "#FF367F"]
        palette = discrete_palette(colors, n=5)[::-1]
        palette_scale = ["Very Low", "Very High"]  # type: ignore[list-item]
        middle_cbar_tick = "Moderate"  # type: ignore[assignment]
        n_digits = 3
    elif layer_type == "spectrum_heatmap":
        colors = ["#FF0000", "#FFFF00", "#008000"]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = values_range  # type: ignore[assignment]
    elif layer_type == "ndvi":
        colors = [
            "#8B0000",
            "#B22222",
            "#DC143C",
            "#FF4500",
            "#FF6347",
            "#FFA500",
            "#FFD700",
            "#FFFF00",
            "#ADFF2F",
            "#32CD32",
            "#008000",
        ]
        palette = linear_gradient(colors, n=256)[::-1]
        palette_scale = [-1, 1]
    elif layer_type == "active_fire":
        colors = ["#ffff00", "#ffaa00", "#ff0000", "#a30119"]
        palette = discrete_palette(colors, n=4)[::-1]
        palette_scale = ["Low confidence Fire", "Very High confidence Fire"]  # type: ignore[list-item]
        middle_cbar_tick = False
    elif layer_type == "rgb_high_res":
        palette_scale = None
        palette = None
        middle_cbar_tick = False
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    return palette_scale, palette, middle_cbar_tick, units, n_digits
