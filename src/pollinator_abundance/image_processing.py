import base64
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
from PIL import Image
import cv2

from pollinator_abundance.math_v2 import create_roi_field_mask
from pollinator_abundance.logconf import create_logger

logger = create_logger(name=__name__)


SCALE_FACTOR = 40000
SCALE_FACTOR_GRAY = 65534
PALETTE_INPUT = [
    (141, 9, 37),
    (255, 188, 110),
    (254, 255, 190),
    (146, 208, 103),
    (5, 94, 51),
]
PALETTE_BLACK_RED_GREEN = [
    (14, 10, 7),
    (229, 37, 31),
    (251, 189, 90),
    (41, 182, 53),
]


def resize_image_pil(image: Image.Image, scale_factor: float) -> Image.Image:
    """Resizes a PIL Image by a given scale factor using bilinear interpolation.

    Args:
        image: The input PIL Image object.
        scale_factor: The factor by which to scale the image dimensions.

    Returns:
        The resized PIL Image object.
    """
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)  # type: ignore[attr-defined]
    return resized_image


def resize_image_x_y(image: Image.Image, ratio_x: float, ratio_y: float) -> Image.Image:
    """Resizes a PIL Image using separate ratios for width and height.

    Args:
        image: The input PIL Image object.
        ratio_x: The factor by which to scale the image width.
        ratio_y: The factor by which to scale the image height.

    Returns:
        The resized PIL Image object using bilinear interpolation.
    """
    new_width = int(image.width * ratio_x)
    new_height = int(image.height * ratio_y)
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)  # type: ignore[attr-defined]
    return resized_image


def resize_image_to_target(array, target_width, target_height):
    if array.shape[0] == target_height and array.shape[1] == target_width:
        return array
    array_temp = np.where(np.isnan(array), -1, array)
    array_resized = cv2.resize(
        array_temp, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    array_resized[array_resized < 0] = np.nan
    return array_resized


def resize_image(array: np.ndarray, reduction_factor: Union[int, float]) -> np.ndarray:
    """
    This function resizes the given matrix by a factor 'reduction_factor'.
    """

    # If reduction_factor == 1 -> nothing to do
    if reduction_factor == 1:
        return array

    # Cells with NaNs are filled with '-1'.
    # As all values are positive, in the final resized image we will replace negative values again with NaNs.
    array_temp = np.where(np.isnan(array), -1, array)

    # Compute reduced dimensions
    height, width = array_temp.shape
    new_height = int(height / reduction_factor)
    new_width = int(width / reduction_factor)

    # Resize the image, by linearly interpolating
    array_resized = cv2.resize(
        array_temp, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # Insert again NaNs
    array_resized[array_resized < 0] = np.nan
    return array_resized


def encode_float_array_to_base64(
    array: np.ndarray,
) -> Tuple[str, Tuple[int, ...], float]:
    """Encodes a float NumPy array into a base64 string.

    The array is scaled to fit within uint16 range (0-SCALE_FACTOR or adjusted
    if max value > 1), converted to uint16, converted to bytes, and then
    base64 encoded.

    Args:
        array: The input NumPy array containing float values.

    Returns:
        A tuple containing:
            - The base64 encoded string representation of the array.
            - The original shape of the input array.
            - The scale factor used for encoding.
    """
    original_shape = array.shape
    max_value = np.nanmax(array)
    if max_value > 1:
        scale_factor = SCALE_FACTOR / max_value
        array_scaled = array * scale_factor
    else:
        scale_factor = SCALE_FACTOR
        array_scaled = np.clip(array, 0, 1) * scale_factor
    array_uint16 = array_scaled.astype(np.uint16)
    binary_data = array_uint16.tobytes()
    encoded_data = base64.b64encode(binary_data).decode()
    return encoded_data, original_shape, scale_factor


def decode_base64_to_float_array(
    encoded_data: str, shape: Tuple[int, ...], scale_factor: float = SCALE_FACTOR
) -> np.ndarray:
    """Decodes a base64 string back into a float NumPy array.

    The base64 string is decoded, interpreted as uint16 bytes, reshaped,
    and scaled back to float values using the provided scale factor.

    Args:
        encoded_data: The base64 encoded string.
        shape: The original shape of the array.
        scale_factor: The scale factor used during encoding. Defaults to SCALE_FACTOR.

    Returns:
        The decoded NumPy array with float32 data type.
    """
    binary_data = base64.b64decode(encoded_data)
    array_uint16 = np.frombuffer(binary_data, dtype=np.uint16)
    return array_uint16.reshape(shape).astype(np.float32) / scale_factor


def interpolate_color(
    color1: Tuple[int, int, int], color2: Tuple[int, int, int], steps: int
) -> List[List[int]]:
    """Linearly interpolates between two RGB colors over a number of steps.

    Generates a list of RGBA colors ([R, G, B, 255]).

    Args:
        color1: The starting RGB color tuple (e.g., (255, 0, 0)).
        color2: The ending RGB color tuple.
        steps: The number of interpolation steps (resulting in 'steps' colors).

    Returns:
        A list of interpolated RGBA colors, where each color is a list
        [R, G, B, 255].
    """
    step_change = [(color2[i] - color1[i]) / steps for i in range(3)]
    return [
        [int(color1[j] + step_change[j] * i) for j in range(3)] + [255]
        for i in range(steps)
    ]


def create_custom_palette(
    colors: List[Tuple[int, int, int]], total_steps: int = 256
) -> List[List[int]]:
    """Creates a custom RGBA color palette by interpolating between given colors.

    Divides the total steps among the transitions between consecutive colors
    and interpolates linearly.

    Args:
        colors: A list of RGB color tuples defining the key points of the palette.
        total_steps: The total number of colors desired in the final palette.
                     Defaults to 256.

    Returns:
        A list of RGBA colors ([R, G, B, 255]), representing the custom palette.
        The list will contain exactly `total_steps` colors.
    """
    num_transitions = len(colors) - 1
    steps_per_transition = total_steps // num_transitions

    palette = []
    for i in range(num_transitions):
        next_steps = (
            steps_per_transition + 1
            if i < total_steps % num_transitions
            else steps_per_transition
        )
        palette += interpolate_color(colors[i], colors[i + 1], next_steps)

    return palette[:total_steps]


def define_palette() -> List[List[int]]:
    """Defines a specific hardcoded palette transitioning from dark red to pine green.

    Generates a 256-color RGBA palette.

    Returns:
        A list of 256 RGBA colors ([R, G, B, 255]).
    """
    palette = []
    for i in range(256):
        # Red component: darker red, inspired by a Bansky-style red
        red = max(255 - i - 70, 0)  # Making red darker
        # Green component: pine green
        green = max(min(20 + i, 255), 0)  # Adjusting green for a pine green shade
        blue = 20  # Keeping blue at 0

        palette.append([red, green, blue, 255])
    return palette


def resize_to_original(
    pa_image_array: np.ndarray, original_image: Image.Image
) -> np.ndarray:
    """Resizes a NumPy array (representing an image) back to the size of an original PIL image.

    Converts the array to a PIL image, resizes it using bilinear interpolation
    to match the dimensions of `original_image`, and converts it back to a
    NumPy array.

    Args:
        pa_image_array: The NumPy array (presumably uint8) to resize.
        original_image: The PIL Image whose size should be matched.

    Returns:
        The resized NumPy array.
    """
    original_size = original_image.size  # (width, height)
    pa_image_pil = Image.fromarray(np.uint8(pa_image_array))
    resized_pa_image_pil = pa_image_pil.resize(original_size, Image.BILINEAR)  # type: ignore[attr-defined]
    resized_pa_image_array = np.array(resized_pa_image_pil)
    return resized_pa_image_array


def create_and_color_image(
    img_array: np.ndarray,
    speed_factor: Union[int, float],  # Allow float reduction factor
    max_val: Union[int, float] = 255,
    palette_input: List[Tuple[int, int, int]] = PALETTE_INPUT,
    min_array_val: Union[int, float] = 0,
    max_array_val: Union[int, float] = 1,
) -> Image.Image:
    """
    Optimized function to create an image from a matrix, coloring values using a palette.
    Uses vectorized operations for significantly better performance.
    """
    # Resize image
    img_array = resize_image(img_array, 1 / speed_factor)

    # Clip the input matrix
    if min_array_val != 0:
        img_array = np.clip(img_array, min_array_val, max_array_val)

    # Normalize the matrix to the RGB range (0-255)
    normalized_img_array = (
        (img_array - min_array_val) / (max_array_val - min_array_val) * max_val
    )
    # Clip the normalized matrix
    normalized_img_array = np.clip(normalized_img_array, 0, 255)
    # Convert data type to 8bit integer
    normalized_img_array = np.uint8(normalized_img_array)  # type: ignore[assignment]

    # Define mask where the output must be transparent
    zero_mask = img_array < min_array_val
    nan_mask = np.isnan(img_array)

    # Create the custom palette
    palette = create_custom_palette(palette_input)

    # Convert Python list palette to NumPy array for efficient indexing
    #    (This could technically be done outside/before this block once)
    palette_array = np.array(palette, dtype=np.uint8)  # Shape should be (256, 4)

    # Perform vectorized palette lookup
    #    Use the uint8 values in normalized_img_array as indices into the palette.
    #    This creates the full colored image according to the palette instantly.
    #    Result shape: (H, W, 4)
    colored_lookup = palette_array[normalized_img_array]

    # Create the combined transparency mask
    #    Combines the zero and NaN conditions.
    #    !! This ASSUMES zero_mask and nan_mask have the same shape as normalized_img_array !!
    transparent_mask = zero_mask | nan_mask

    # Apply the transparency mask to the looked-up colors
    #    Where the mask is True, set the alpha channel (index 3) to 0.
    #    We can modify colored_lookup directly or assign to colored_array. Let's modify.
    colored_lookup[transparent_mask, 3] = 0

    # Initialize the output
    colored_array = np.zeros((*normalized_img_array.shape, 4), dtype=np.uint8)

    # Assign the result to the pre-initialized output array
    colored_array[:, :, :] = colored_lookup

    # Create the image and return it
    result_image_rgba = Image.fromarray(colored_array, "RGBA")
    return result_image_rgba


def create_gray_image(
    img_array: np.ndarray, speed_factor: float, molt: int = SCALE_FACTOR_GRAY
) -> Image.Image:
    """Creates a grayscale 16-bit integer image from a NumPy array.

    The input array is resized based on `speed_factor`, scaled by `molt`,
    offset by 1, clipped to the uint16 range [0, 65535], and converted
    to a uint16 array. This array is then used to create a PIL Image
    in "I;16" mode (16-bit grayscale).

    Args:
        img_array: The input NumPy array with data values.
        speed_factor: Factor to resize the input array (1 / speed_factor is used
                      in resize_image, so speed_factor > 1 enlarges).
        molt: The multiplication factor used for scaling before clipping.
              Defaults to SCALE_FACTOR_GRAY.

    Returns:
        A PIL Image object in "I;16" mode (16-bit grayscale).
    """
    img_array = resize_image(img_array, 1 / speed_factor)
    img_array_plus_one = img_array * molt + 1
    normalized_img_array = np.clip(img_array_plus_one, 0, 65535)
    normalized_img_array = np.uint16(normalized_img_array)  # type: ignore[assignment]
    result_image_gray = Image.fromarray(normalized_img_array, "I;16")
    # print(result_image_gray.mode)
    return result_image_gray


def merge_roi_an_ca_image(
    image_roi: Image.Image, image_ca: Image.Image, align_x: float, align_y: float
) -> Image.Image:
    """Merges an ROI image onto a CA image at specified alignment coordinates.

    Makes black pixels in the ROI image transparent before pasting it onto
    the CA image. Assumes ROI image dimensions are less than or equal to
    CA image dimensions.

    Args:
        image_roi: The PIL Image representing the Region of Interest (ROI).
                   Should have RGBA mode potentially.
        image_ca: The PIL Image representing the Context Area (CA).
        align_x: The x-coordinate (left offset) on the CA image where the
                 top-left corner of the ROI image will be pasted.
        align_y: The y-coordinate (top offset) on the CA image where the
                 top-left corner of the ROI image will be pasted.

    Returns:
        The CA image with the ROI image pasted onto it.

    Raises:
        ValueError: If the ROI image dimensions are larger than the CA image dimensions.
    """
    # print(image_roi.size, image_ca.size)
    if image_roi.size > image_ca.size:
        raise ValueError("Roi è maggiore di CA come dimensione dell'immagine")

    width, height = image_roi.size
    new_image = Image.new("RGBA", (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b, a = image_roi.getpixel((x, y))  # type: ignore[misc]
            # Se il pixel è nero (o quasi nero), imposta l'alfa a 0
            if r < 2 and g < 2 and b < 2:  # puoi regolare questi valori se necessario
                new_image.putpixel((x, y), (r, g, b, 0))
            else:
                new_image.putpixel((x, y), (r, g, b, a))

    image_ca.paste(new_image, (int(align_x), int(align_y)), new_image)
    return image_ca


def merge_roi_an_ca_array(
    image_roi: np.ndarray, image_ca: np.ndarray, align_x: float, align_y: float
) -> np.ndarray:
    """Merges an ROI NumPy array onto a CA NumPy array at specified coordinates.

    Overwrites values in the CA array with corresponding non-negative values
    from the ROI array, positioned according to the alignment points.

    Args:
        image_roi: The NumPy array representing the Region of Interest (ROI).
        image_ca: The NumPy array representing the Context Area (CA).
        align_x: The starting column index in the CA array for the merge.
        align_y: The starting row index in the CA array for the merge.

    Returns:
        The modified CA NumPy array with the ROI array merged onto it.

    Raises:
        ValueError: If ROI array dimensions are larger than CA array dimensions.
        ValueError: If the ROI placement based on alignment points extends
                    beyond the bounds of the CA array.
    """
    # Check that ROI is smaller than CA
    if image_roi.shape > image_ca.shape:
        raise ValueError("ROI image dimensions are larger than CA image ones")
    # Compute ending coordinates
    height, width = image_roi.shape
    end_y = int(align_y) + height
    end_x = int(align_x) + width
    if end_y > image_ca.shape[0] or end_x > image_ca.shape[1]:
        raise ValueError("Ending coordinates fall outside CA dimension")
    # Identify white pixels in image_roi
    mask = image_roi >= 0
    # Merge arrays
    image_ca[int(align_y) : end_y, int(align_x) : end_x][mask] = image_roi[mask]
    return image_ca


def find_bounding_box(
    polygon: Union[List[List[float]], np.ndarray], padding: float
) -> Tuple[float, float, float, float]:
    """Computes the bounding box for a polygon with optional padding.

    Args:
        polygon: A list of [x, y] coordinates or a NumPy array defining the
                 vertices of the polygon.
        padding: A value to add/subtract to the min/max coordinates to pad
                 the bounding box.

    Returns:
        A tuple containing (min_x, min_y, max_x, max_y) of the padded
        bounding box.
    """
    polygon = np.array(polygon)
    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])
    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


def inverse_pa_rgba(
    value_rgba: Union[List[int], Tuple[int, ...]],
    max_val: float = 255 * 2.5,
    palette_input: List[Tuple[int, int, int]] = PALETTE_INPUT,
    min_array_val: float = 0.0,
    max_array_val: float = 1.0,
) -> float:
    """Approximates the original data value from an RGBA color using a palette map.

    Creates a reverse mapping from RGB colors (derived from `palette_input`)
    back to an index (0-255). This index is then scaled back to the original
    data range [`min_array_val`, `max_array_val`]. Black input pixels (0,0,0)
    are mapped to NaN. Colors not found exactly in the palette are mapped to NaN
    (assuming the index lookup defaults to 255 which is then treated as NaN).

    Args:
        value_rgba: A list or tuple representing the RGBA color (e.g., [R, G, B, A]).
                    Only the first three (RGB) components are used.
        max_val: The scaling factor used in the reverse transformation. Check if
                 this corresponds to the `max_val` used in `create_and_color_image`.
                 Defaults to 255 * 2.5.
        palette_input: List of RGB tuples used to generate the original palette.
                       Defaults to PALETTE_INPUT.
        min_array_val: The minimum value of the original data range. Defaults to 0.
        max_array_val: The maximum value of the original data range. Defaults to 1.

    Returns:
        The estimated original float value, or np.nan if the color corresponds
        to the zero mask or is not found in the palette.
    """
    # Create the custom palette
    palette = create_custom_palette(palette_input)

    # Reverse the palette mapping
    palette_map = {tuple(color[:3]): idx for idx, color in enumerate(palette)}

    rgba = tuple(value_rgba[:3])

    if rgba == (0, 0, 0):  # Transparent or zero mask
        out = np.nan  # Assuming np.nan was used for NaNs
    else:
        pixel_value = float(palette_map.get(rgba, 255))
        if pixel_value == 255:
            out = np.nan
        else:
            out = (pixel_value / max_val) * (
                max_array_val - min_array_val
            ) + min_array_val

    return out


def inverse_msa_rgba(
    img_rgba: np.ndarray,
    palette_input: List[Tuple[int, int, int]] = PALETTE_BLACK_RED_GREEN,
    max_val: float = 255.0,
    min_array_val: float = 0.0,
    max_array_val: float = 1.0,
) -> np.ndarray:
    """Converts an RGBA/RGB image array back to original scalar values (e.g., MSA).

    Uses a reverse color lookup based on the provided palette definition. The
    resulting palette index (0-255, or `max_norm_val`) is then scaled to the original data range
    defined by `min_array_val` and `max_array_val`. Pixels with alpha=0 (if RGBA)
    or colors not found in the palette map are converted to NaN.

    This function is vectorized for efficient processing of NumPy arrays.

    Args:
        img_rgba: Input NumPy array representing the image. Must have shape
                  (..., 3) for RGB or (..., 4) for RGBA.
        palette_input: List of RGB tuples used to generate the color palette for
                       the reverse lookup. Defaults to `PALETTE_BLACK_RED_GREEN`.
        max_norm_val: The maximum value of the normalized palette index range
                      used for scaling (typically 255.0). Defaults to 255.0.
        min_array_val: The minimum value of the original data range to scale back to.
                       Defaults to 0.0.
        max_array_val: The maximum value of the original data range to scale back to.
                       Defaults to 1.0.

    Returns:
        A NumPy array of floats with the same leading dimensions as the input
        array, containing the calculated original scalar values. Unmapped or
        transparent pixels are represented as np.nan.

    Raises:
        ValueError: If the input image array does not have 3 or 4 channels in
                    its last dimension.
    """
    # Create the custom palette
    palette = create_custom_palette(palette_input)

    # Reverse the palette mapping
    palette_map = {tuple(color[:3]): idx for idx, color in enumerate(palette)}

    # Check number of channels
    num_channels = img_rgba.shape[-1]
    if num_channels not in [3, 4]:
        raise ValueError("Input image must have either 3 (RGB) or 4 (RGBA) channels.")

    # Extract the RGB part of the image
    rgb_values = img_rgba[..., :3]

    # Define a vectorized function to map RGB tuples to palette indices
    vectorized_mapping = np.vectorize(
        lambda r, g, b: palette_map.get((r, g, b), 255), otypes=[np.float32]
    )

    # Apply the vectorized mapping to the RGB values
    pixel_values = vectorized_mapping(
        rgb_values[..., 0], rgb_values[..., 1], rgb_values[..., 2]
    )

    # Scale to original values
    scaled_values: np.ndarray = (pixel_values / max_val) * (
        max_array_val - min_array_val
    ) + min_array_val

    # Handle unmapped pixels or transparency if alpha channel is present
    if num_channels == 4:
        transparency_mask = img_rgba[..., 3] == 0
        scaled_values[transparency_mask] = np.nan

    return scaled_values


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Converts geographic coordinates (latitude, longitude) to tile numbers (x, y)
    for a given zoom level using the Web Mercator projection.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        zoom: The map zoom level.

    Returns:
        A tuple containing the (xtile, ytile) numbers.
    """
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Converts Web Mercator tile numbers (x, y) at a given zoom level back to
    geographic coordinates (latitude, longitude) of the tile's top-left corner.

    Args:
        xtile: The x tile number.
        ytile: The y tile number.
        zoom: The map zoom level.

    Returns:
        A tuple containing the (latitude, longitude) in degrees.
    """
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_map_image(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int
) -> Tuple[Image.Image, float, float, float, float]:
    """Downloads and stitches map tiles to create a map image for the given bounding box.

    Downloads tiles from ArcGIS World Imagery service covering the specified
    latitude/longitude bounds at the given zoom level, stitches them together,
    and crops the resulting image precisely to the requested bounds.

    Args:
        min_lat: Minimum latitude of the bounding box.
        min_lon: Minimum longitude of the bounding box.
        max_lat: Maximum latitude of the bounding box.
        max_lon: Maximum longitude of the bounding box.
        zoom: The map zoom level for tile download.

    Returns:
        A tuple containing:
            - map_img: A PIL Image object of the stitched and cropped map.
            - lat_top: The northernmost latitude of the final cropped image (max_lat).
            - lon_left: The westernmost longitude of the final cropped image (min_lon).
            - lat_bottom: The southernmost latitude of the final cropped image (min_lat).
            - lon_right: The easternmost longitude of the final cropped image (max_lon).
    """
    # Ensure min values are less than max values
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    # Calculate tile numbers
    x_min, y_max = deg2num(max_lat, min_lon, zoom)
    x_max, y_min = deg2num(min_lat, max_lon, zoom)

    # Swap if necessary
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    # Number of tiles to download
    x_tiles = x_max - x_min + 1
    y_tiles = y_max - y_min + 1

    # Create a new image
    tile_size = 256
    map_width = x_tiles * tile_size
    map_height = y_tiles * tile_size
    map_img = Image.new("RGB", (map_width, map_height))

    # Download and paste tiles
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            response = requests.get(tile_url)
            if response.status_code == 200:
                tile_img = Image.open(BytesIO(response.content))
                x_offset = (x - x_min) * tile_size
                y_offset = (y - y_min) * tile_size
                map_img.paste(tile_img, (x_offset, y_offset))
            else:
                print(f"Failed to download tile {x}, {y}")

    # Calculate the geographic extent
    lat_top, lon_left = num2deg(x_min, y_min, zoom)
    lat_bottom, lon_right = num2deg(x_max + 1, y_max + 1, zoom)

    # Calculate pixel offsets for cropping
    lat_range = lat_top - lat_bottom
    lon_range = lon_right - lon_left

    top_pixel = int((lat_top - max_lat) / lat_range * map_height)
    bottom_pixel = int((lat_top - min_lat) / lat_range * map_height)
    left_pixel = int((min_lon - lon_left) / lon_range * map_width)
    right_pixel = int((max_lon - lon_left) / lon_range * map_width)

    # Crop the image to the exact bounds
    map_img = map_img.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))

    # Update the geographic extent after cropping
    lat_top = max_lat
    lat_bottom = min_lat
    lon_left = min_lon
    lon_right = max_lon

    return map_img, lat_top, lon_left, lat_bottom, lon_right


def inner_check_layer_colors(image_bytes: bytes, areas: Dict[str, Any]) -> None:
    """Checks if colors in an image match the colors defined in area metadata.

    Opens an image from bytes, extracts all unique pixel colors (RGB), and
    compares them against a set of expected colors derived from the `areas`
    dictionary (plus black).

    Args:
        image_bytes: The image data as bytes.
        areas: A dictionary where values are dictionaries, each containing at
               least a 'color' key with a hex string (e.g., "#RRGGBB").

    Raises:
        ValueError: If any color found in the image pixels is not present in the
                    set of expected colors derived from `areas` (or black).
    """
    image_filelike = BytesIO(image_bytes)
    img = Image.open(image_filelike)
    black = (0, 0, 0)
    colors = {tuple(int(a["color"][i : i + 2], 16) for i in (1, 3, 5)) for a in areas}  # type: ignore[index]
    colors.add(black)

    image_colors = set(img.getdata())
    image_colors = {(p[0], p[1], p[2]) for p in image_colors}

    if not image_colors.issubset(colors):
        raise ValueError(
            "Discrepancy between colors in the image and colors of the areas"
        )


def geo_to_pixel(
    lat: float,
    lon: float,
    lat_top: float,
    lon_left: float,
    lat_bottom: float,
    lon_right: float,
    map_width: int,
    map_height: int,
) -> Tuple[int, int]:
    """Converts geographic coordinates (latitude, longitude) to pixel coordinates (x, y)
    within a given map image bounds and dimensions.

    Assumes a linear mapping between geographic coordinates and pixel coordinates.

    Args:
        lat: Latitude of the point to convert.
        lon: Longitude of the point to convert.
        lat_top: The northernmost latitude of the map image.
        lon_left: The westernmost longitude of the map image.
        lat_bottom: The southernmost latitude of the map image.
        lon_right: The easternmost longitude of the map image.
        map_width: Width of the map image in pixels.
        map_height: Height of the map image in pixels.

    Returns:
        A tuple containing the (x, y) pixel coordinates corresponding to the
        input latitude and longitude.
    """
    x = int((lon - lon_left) / (lon_right - lon_left) * map_width)
    y = int((lat_top - lat) / (lat_top - lat_bottom) * map_height)
    return (x, y)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a HEX color string (e.g., '#RRGGBB' or 'RRGGBB') to an RGB tuple.

    Args:
        hex_color: The color string in hexadecimal format.

    Returns:
        A tuple containing the (R, G, B) integer values (0-255).
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
    """Converts an RGB tuple (e.g., (255, 0, 0)) to a HEX color string ('#RRGGBB').

    Args:
        rgb_tuple: A tuple containing the (R, G, B) integer values (0-255).

    Returns:
        The color string in hexadecimal format, starting with '#'.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)


def linear_gradient(
    colors: Sequence[Union[str, Tuple[int, int, int]]], n: int = 256
) -> List[Tuple[int, int, int]]:
    """Generates a list of colors forming a linear gradient between specified colors.

    Interpolates linearly between consecutive colors in the input list to
    produce a gradient with a total of `n` colors.

    Args:
        colors: A list of key colors for the gradient. Each color can be a
                HEX string (e.g., '#FF0000') or an RGB tuple (e.g., (255, 0, 0)).
                Requires at least two colors.
        n: The total number of colors to generate in the gradient. Defaults to 256.

    Returns:
        A list of `n` RGB tuples representing the gradient.

    Raises:
        ValueError: If less than two colors are provided in the `colors` list.
        ValueError: If an invalid color format is encountered in the `colors` list.
    """

    # Convert all colors to RGB tuples
    rgb_colors = []
    for color in colors:
        if isinstance(color, str):
            rgb_colors.append(hex_to_rgb(color))
        elif isinstance(color, tuple) and len(color) == 3:
            rgb_colors.append(color)
        else:
            raise ValueError(f"Invalid color format: {color}")

    # Number of segments
    segments = len(rgb_colors) - 1
    if segments == 0:
        raise ValueError("At least two colors are required for a gradient.")

    # Initialize gradient list
    gradient = []

    # Calculate the number of colors per segment
    colors_per_segment = n / segments

    for i in range(segments):
        start_color = np.array(rgb_colors[i])
        end_color = np.array(rgb_colors[i + 1])

        # Determine start and end indices for interpolation
        start_index = int(round(i * colors_per_segment))
        end_index = int(round((i + 1) * colors_per_segment))

        # Handle the last segment to include all remaining colors
        if i == segments - 1:
            end_index = n

        # Generate interpolated colors for the current segment
        for t in np.linspace(0, 1, end_index - start_index, endpoint=False):
            interpolated = (1 - t) * start_color + t * end_color
            gradient.append(tuple(interpolated.astype(int)))

    # Append the last color to complete the gradient
    gradient.append(rgb_colors[-1])

    return gradient


def discrete_palette(
    colors: Sequence[Union[str, Tuple[int, int, int]]], n: Optional[int] = None
) -> List[Tuple[int, int, int]]:
    """Selects or returns discrete colors from a provided list.

    If `n` is None, returns all provided colors converted to RGB tuples.
    If `n` is specified, selects `n` colors approximately evenly spaced
    from the input list.

    Args:
        colors: A list of colors. Each color can be a HEX string or an RGB tuple.
        n: The number of discrete colors to select. If None, all colors are used.
           Defaults to None.

    Returns:
        A list of selected/converted RGB tuples.

    Raises:
        ValueError: If `n` is greater than the number of provided colors.
    """
    if n is None:
        return [
            hex_to_rgb(color) if isinstance(color, str) else color for color in colors
        ]
    else:
        if n > len(colors):
            raise ValueError("n cannot be greater than the number of provided colors.")
        indices = np.linspace(0, len(colors) - 1, n).astype(int)
        return [
            hex_to_rgb(colors[i]) if isinstance(colors[i], str) else colors[i]
            for i in indices
        ]


def jet_colormap(n: int = 256) -> List[Tuple[int, int, int]]:
    """Generates the Jet colormap with a specified number of colors.

    Implements the standard Jet colormap algorithm.

    Args:
        n: The number of colors desired in the colormap. Defaults to 256.

    Returns:
        A list of `n` RGB tuples representing the Jet colormap.
    """
    palette = []
    for i in range(n):
        x = i / (n - 1)  # Normalize to [0,1]
        r = 1.5 - abs(4 * x - 3)
        g = -4 * x + 4
        b = -1.5 + abs(4 * x - 1)

        # Clip the values to [0,1]
        r = max(0.0, min(r, 1.0))
        g = max(0.0, min(g, 1.0))
        b = max(0.0, min(b, 1.0))

        # Convert to 0-255 scale
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        palette.append((r, g, b))
    return palette


def generate_roi_and_ca_mask(
    array_pn: np.ndarray,
    site_pixel_polygons: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates binary masks distinguishing the ROI and CA areas.

    Creates a mask for the ROI based on either site polygons (`roi_mask_origin`='polygon')
    or a previous version of the ROI (`roi_mask_origin`='v1'). The CA mask is
    the inverse of the ROI mask, excluding NaN areas in the combined array.
    Requires several imported functions for data retrieval and processing.

    Args:
        roi_mask_origin: Method to generate ROI mask ('polygon' or 'v1').
        roi_id: Identifier for the current ROI. Used for comparison if 'v1' is chosen.
        array_pn: Combined NumPy array (ROI merged onto CA) used for NaN masking
                  and potentially for polygon masking.
        array_pn_ca: NumPy array for the Context Area, used as a base size if 'v1' method is used.
        site_pixel_polygons: Polygon data used if `roi_mask_origin` is 'polygon'.
        alignment_point_x: X-coordinate for aligning ROI if 'v1' method is used.
        alignment_point_y: Y-coordinate for aligning ROI if 'v1' method is used.
        plantations_polygons_id: Identifier to retrieve the first ROI version if 'v1' is used.
        s3_bucket: Name of the S3 bucket to read images from if 'v1' method is used.

    Returns:
        A tuple containing:
            - mask_roi_field: A binary NumPy array (0 or 1) where 1 indicates the ROI field.
            - mask_ca: A binary NumPy array (0 or 1) where 1 indicates the CA field (excluding NaN areas).
    """
    mask_roi_field = create_roi_field_mask(array_pn, site_pixel_polygons)

    mask_ca = 1 - mask_roi_field
    nan_mask = np.isnan(array_pn)
    mask_ca[nan_mask] = 0

    return mask_roi_field, mask_ca


def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Applies a binary mask to the alpha channel of a PIL Image.

    Converts the input image to RGBA if it isn't already. Handles grayscale
    input images by coloring them first using `create_and_color_image` with
    a default gradient. Multiplies the image's alpha channel by the mask values
    (assuming 0 for masked out, 1 for kept).

    Args:
        image: The input PIL Image.
        mask: A NumPy array (binary, 0 or 1) of the same height and width as
              the image, used to mask the alpha channel.

    Returns:
        A new PIL Image object (RGBA) with the mask applied to its alpha channel.
    """
    # Convert the image (Pillow) to a NumPy array
    image_array = np.array(image)
    # If image is not RGBA, convert it. This happens only for FA maps (black and white).
    if len(image_array.shape) == 2:
        # Normalize the image
        image_array = image_array / image_array.max()
        # Color the image
        colored_image = create_and_color_image(
            img_array=image_array,
            speed_factor=1,
            max_val=255,
            palette_input=linear_gradient(PALETTE_INPUT, n=256),
        )
        # Now the image is RGBA
        image_array = np.array(colored_image)
    # If the image has no alpha channel, add it
    if image_array.shape[2] == 3:
        image_array = np.dstack([image_array, np.ones_like(image_array[:, :, 0]) * 255])
    # Apply the mask (assuming a binary mask: 0 for masking, 1 for keeping)
    image_array[:, :, -1] *= mask.astype(image_array.dtype)
    # Convert back to a Pillow image
    masked_image = Image.fromarray(image_array)
    return masked_image
