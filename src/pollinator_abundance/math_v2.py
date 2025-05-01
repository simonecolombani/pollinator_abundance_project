import math

# from math_v1 import *

import numpy as np
import upolygon

from pollinator_abundance.math_v1 import crop_to_same_size, pa_multiply


def create_roi_field_mask(
    array_image: np.ndarray, site_pixel_polygon: list[np.ndarray]
) -> np.ndarray:
    """
    Creates a binary mask for a region of interest (ROI) defined by polygons,
    using the upolygon library.

    This function replicates the behavior of using cv2.fillPoly to create a mask.

    Args:
        array_image (np.ndarray): The reference image NumPy array, used to determine
                                   the shape of the output mask.
        site_pixel_polygon (list[np.ndarray]): A list of polygons defining the ROI.
                                     Each polygon should be a NumPy array of shape
                                     (N, 2) or (N, 1, 2) containing integer vertex
                                     coordinates, similar to the format accepted by
                                     cv2.fillPoly.

    Returns:
        np.ndarray: A binary mask (dtype=np.uint8) with the same shape as
                    array_image, where pixels inside the specified polygons are
                    set to 1 and others are 0.
    """
    if array_image is None:
        raise ValueError("Input 'array_image' cannot be None.")
    if site_pixel_polygon is None or not isinstance(site_pixel_polygon, list):
        raise ValueError("'site_pixel_polygon' must be a list of NumPy arrays.")

    # Initialize mask with the target shape but using int32 for upolygon compatibility
    mask_shape = array_image.shape[:2]  # Use only H, W for the mask shape
    mask = np.zeros(mask_shape, dtype=np.int32)
    fill_value = 1  # The value to fill the polygon with (like the original function)

    # Convert polygon vertices to the flat format expected by upolygon
    # upolygon expects a list of lists/arrays like [[x1, y1, x2, y2,...], [x1, y1,...]]
    upolygon_paths = []
    for i, poly_cv2 in enumerate(site_pixel_polygon):
        if not isinstance(poly_cv2, np.ndarray):
            print(
                f"Warning: Item {i} in site_pixel_polygon is not a NumPy array, skipping."
            )
            continue

        # Ensure the polygon array has 2 or 3 dimensions (like (N,2) or (N,1,2))
        if poly_cv2.ndim < 2 or poly_cv2.ndim > 3:
            print(
                f"Warning: Polygon {i} has unexpected dimensions {poly_cv2.ndim}, skipping."
            )
            continue
        # Ensure the last dimension has size 2 (x, y coordinates)
        if poly_cv2.shape[-1] != 2:
            print(
                f"Warning: Polygon {i} vertices do not seem to be (x, y) pairs (shape: {poly_cv2.shape}), skipping."
            )
            continue

        # Reshape to (N, 2) to handle both (N, 2) and (N, 1, 2) inputs
        num_vertices = poly_cv2.shape[-2]
        try:
            poly_np = poly_cv2.reshape(num_vertices, 2)
        except ValueError as e:
            print(
                f"Warning: Could not reshape polygon {i} (shape: {poly_cv2.shape}) to (N, 2): {e}, skipping."
            )
            continue

        # Need at least 3 vertices to form a polygon
        if num_vertices < 3:
            print(
                f"Warning: Polygon {i} has less than 3 vertices ({num_vertices}), skipping."
            )
            continue

        # Ensure vertices are integers and flatten
        # upolygon expects integer coordinates
        flat_vertices = poly_np.astype(np.int32).ravel().tolist()
        upolygon_paths.append(flat_vertices)

    # If there are valid polygons to draw, call upolygon
    if upolygon_paths:
        try:
            upolygon.draw_polygon(mask, upolygon_paths, fill_value)
        except Exception as e:
            # Log the error or raise it depending on desired behavior
            print(f"Error calling upolygon.draw_polygon: {e}")
            # Optionally raise an error or return the empty mask
            # raise  # Uncomment to propagate the error
            return np.zeros(
                mask_shape, dtype=np.uint8
            )  # Return empty uint8 mask on error
    else:
        print("Warning: No valid polygons found to draw.")

    # Cast the final mask to uint8 to match the original function's output type
    return mask.astype(np.uint8)


def calculateNectarPotential(clc_values):
    nectarpotential = 0
    ha = 0
    for clc in clc_values:
        if clc["pn_mean"] is not None:
            nectarpotential += clc["hectare"] * clc["pn_mean"]
            ha += clc["hectare"]
    return nectarpotential / ha


def encode_rgb_to_hex_fast(img_array):
    """
    This function converts RGB values matrix into hex integers matrix (mapping each color into an integer)
    """
    # Pack RGB values into a single integer, treating each color channel as an 8-bit component
    hex_encoded = (
        img_array[..., 0].astype(int) * 65536
        + img_array[..., 1].astype(int) * 256
        + img_array[..., 2].astype(int)
    )
    return hex_encoded


def map_hex_to_values(hex_encoded, color_to_ns):
    """
    This function converts a color-matrix (where colors have been mapped into integers) to values-matrix, using given color-value dictionary.
    """
    # Convert color keys from hex string to integer
    color_keys = {int(key[1:], 16): value for key, value in color_to_ns.items()}
    # Initialize output array
    result = np.full(hex_encoded.shape, np.nan)  # Use None or another default value
    # Vectorized lookup
    for hex_value, ns_value in color_keys.items():
        result[hex_encoded == hex_value] = ns_value
    return result


def image_to_clc_ns_v3(image, clc_table, ns_name):
    """
    This function converts the given image from color-scale to K-value-scale, where K is the parameters whose name
    is given through the 'ns_name' input.
    The input is an image, the output is a NumPy matrix.
    """
    # Convert image to matrix
    img_array = np.array(image)
    # Create color-value dictionary
    color_to_ns = {row["color"].lower(): row[ns_name] for row in clc_table}
    color_to_ns["#000000"] = np.nan
    if "#ffffff" not in color_to_ns:
        color_to_ns["#ffffff"] = np.nan
    # Convert RGB to encoded hex integers
    hex_encoded = encode_rgb_to_hex_fast(img_array)
    # Convert color_matrix to values matrix, using color-value dictionary
    img_clc_ns = map_hex_to_values(hex_encoded, color_to_ns)
    return img_clc_ns


def process_pixel_block_32bit(img_array, ratio_x, ratio_y, alfa, i_start, i_end):
    """
    This function performs the following steps, for each cell of the given matrix where i_start <= row index < i_end:
        - it computes the distance of this cell with respect to all the other cells of the matrix;
        - it converts the distances in weights (using exponential decay)
        - it computes the weighted mean of the matrix using these weights
    Thus the output of this function is a matrix where each cell contains the value computed as described.
    """
    # Get image height and width
    height, width = img_array.shape
    # Define grid of coordinates
    x_coords, y_coords = np.meshgrid(
        np.arange(width) * ratio_x, np.arange(height) * ratio_y
    )
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)
    # Initialize result
    partial_result = np.full(
        (min(i_end, height) - i_start, width), np.nan, dtype=np.float32
    )
    # Compute NaN mask
    nan_mask = np.isnan(img_array)
    # Mask NaNs in the matrix
    img_array_masked = np.where(nan_mask, 0, img_array)
    # Iterate over rows with index between i_start and min(i_end, height)
    for i in range(i_start, min(i_end, height)):
        # Iterate over columns
        for j in range(width):
            # Skip NaNs
            if nan_mask[i, j]:
                continue
            # Compute distances for current pixel
            distances = np.sqrt(
                (x_coords - x_coords[i, j]) ** 2 + (y_coords - y_coords[i, j]) ** 2
            )
            # Compute exponential weights
            weights = np.exp(-distances / alfa)
            # Apply NaN mask
            weights[nan_mask] = 0
            # Compute weighted sum and total weight
            weighted_sum = np.sum(img_array_masked * weights)
            total_weight = np.sum(weights)
            # Compute weighted mean for current pixel
            if total_weight != 0:
                partial_result[i - i_start, j] = weighted_sum / total_weight
    return partial_result


def fill_nans_with_neighbors(img_array: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Fills NaN values in a NumPy array with the mean of their valid (non-NaN) neighbors.
    Uses a summed-area table approach for efficient calculation without SciPy.
    Args:
        img_array: The input NumPy array (e.g., an image) potentially containing NaNs.
        window_size: The size of the square window (must be an odd integer >= 1).
                     Defaults to 3 (for a 3x3 window).
    Returns:
        A new NumPy array with NaN values filled, or the original array if no NaNs
        were present or if window_size is invalid. Original NaNs that have no
        valid neighbors in their window remain NaN.
    """
    # --- Input Validation ---
    if not isinstance(img_array, np.ndarray):
        raise TypeError("Input 'img_array' must be a NumPy array.")
    if not isinstance(window_size, int) or window_size < 1 or window_size % 2 == 0:
        print(
            f"Warning: window_size must be a positive odd integer. Received {window_size}. Returning original array."
        )
        return img_array.copy()  # Return a copy to maintain consistency

    # --- Handle No-NaN Case ---
    nan_mask = np.isnan(img_array)
    if not np.any(nan_mask):
        # No NaNs found, return a copy of the original array
        return img_array.copy()

    # --- Prepare Arrays for Summed-Area Table Calculation ---
    rows, cols = img_array.shape
    p = window_size // 2  # Padding size

    # Create an array where NaNs are 0, for summation purposes
    img_zeros = np.nan_to_num(img_array, nan=0.0)
    # Create a mask where valid numbers are 1 and NaNs are 0, for counting purposes
    valid_mask = (~nan_mask).astype(img_array.dtype)

    # Pad arrays with zeros to handle borders during window summation
    # Padded size will be (rows + 2*p, cols + 2*p)
    img_zeros_padded = np.pad(
        img_zeros, pad_width=p, mode="constant", constant_values=0
    )
    valid_mask_padded = np.pad(
        valid_mask, pad_width=p, mode="constant", constant_values=0
    )

    # --- Calculate Summed-Area Tables (SAT) ---
    # SAT allows calculating the sum of any rectangular area in O(1) time
    sat_sum = np.cumsum(np.cumsum(img_zeros_padded, axis=0), axis=1)
    sat_count = np.cumsum(np.cumsum(valid_mask_padded, axis=0), axis=1)

    # --- Calculate Neighbor Sum and Count using SAT ---
    # Add a row/column of zeros at the top/left of SATs for easier window calculation
    # New shape: (rows + 2*p + 1, cols + 2*p + 1)
    sat_sum_padded = np.pad(
        sat_sum, pad_width=((1, 0), (1, 0)), mode="constant", constant_values=0
    )
    sat_count_padded = np.pad(
        sat_count, pad_width=((1, 0), (1, 0)), mode="constant", constant_values=0
    )

    # Calculate the sum and count within each window using the SAT properties
    # The indices correspond to the bottom-right corner of the window in the padded array
    # We use slicing to perform this calculation efficiently for all pixels
    w = window_size  # Full window width
    # Indices explanation (using sat_sum_padded as example):
    # sat_sum_padded[w:, w:] -> bottom-right corners
    # sat_sum_padded[:-w, w:] -> top-right corners (shifted)
    # sat_sum_padded[w:, :-w] -> bottom-left corners (shifted)
    # sat_sum_padded[:-w, :-w] -> top-left corners (shifted)
    # This implements the inclusion-exclusion principle for window sums
    neighbor_sum = (
        sat_sum_padded[w:, w:]
        - sat_sum_padded[:-w, w:]
        - sat_sum_padded[w:, :-w]
        + sat_sum_padded[:-w, :-w]
    )
    neighbor_count = (
        sat_count_padded[w:, w:]
        - sat_count_padded[:-w, w:]
        - sat_count_padded[w:, :-w]
        + sat_count_padded[:-w, :-w]
    )

    # Ensure shapes match the original image array
    assert neighbor_sum.shape == (rows, cols)
    assert neighbor_count.shape == (rows, cols)

    # --- Calculate Mean and Fill NaNs ---
    # Initialize the result array as a copy of the original
    img_filled = img_array.copy()

    # Identify locations of original NaNs where there are valid neighbors
    # i.e., where neighbor_count > 0
    fill_mask = nan_mask & (neighbor_count > 0)

    # Calculate the mean only where the count is positive
    # Using np.divide handles division by zero implicitly by not writing where count is 0
    # but we explicitly use the fill_mask for clarity and safety.
    valid_means = np.divide(neighbor_sum[fill_mask], neighbor_count[fill_mask])

    # Fill the identified NaN locations with the calculated means
    img_filled[fill_mask] = valid_means

    # NaNs where neighbor_count was 0 will remain NaN
    return img_filled


def pixel_mean_calculation_nan_optimized_2D_32bit(
    img_array: np.ndarray, alfa: float, ratio_x: float, ratio_y: float
):
    """
    This function performs the following steps, for each cell of the given matrix:
        - it computes the distance of this cell with respect to all the other cells of the matrix;
        - it converts the distances in weights (using exponential decay)
        - it computes the weighted mean of the matrix using these weights
    Thus the output of this function is a matrix where each cell contains the value computed as described.
    """
    # Get image height and width
    height, width = img_array.shape
    # Define grid of coordinates
    x_coords, y_coords = np.meshgrid(
        np.arange(width) * ratio_x, np.arange(height) * ratio_y
    )
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)
    # Initialize result
    img_result = np.full((height, width), np.nan, dtype=np.float32)
    # Compute NaN mask
    nan_mask = np.isnan(img_array)
    not_nan_mask = ~nan_mask
    # Mask NaNs in the matrix
    img_array_masked = np.where(nan_mask, 0, img_array)

    # NOTE: Vector computation. Compute not-nan vectors out of the loop
    x_coords_vec = x_coords[not_nan_mask]
    y_coords_vec = y_coords[not_nan_mask]
    img_array_masked_vec = img_array_masked[not_nan_mask]

    # Iterate over rows
    for i in range(height):
        # Iterate over columns
        for j in range(width):
            # Skip NaNs
            if nan_mask[i, j]:
                continue

            # --- NEW ---
            # NOTE: Compute the not-nan vector for all N x M matrices and perform the same OLD
            # operations on 1 x K vectors (where K is the number of not-nan pixels). This removes
            # the need to compute the weight on all pixels which do not contribute anyway to the
            # final sum.
            # Compute distances for current pixel
            distances_vec = np.sqrt(
                (x_coords_vec - x_coords[i, j]) ** 2
                + (y_coords_vec - y_coords[i, j]) ** 2
            )
            # Compute exponential weights
            weights_vec = np.exp(-distances_vec / alfa)
            # Compute weighted sum and total weight
            weighted_sum = np.sum(img_array_masked_vec * weights_vec)
            total_weight = np.sum(weights_vec)
            # -----------

            # Compute weighted mean for current pixel
            if total_weight != 0:
                img_result[i, j] = weighted_sum / total_weight
            else:
                # If no weight is applied, handle NaN result by filling with a fallback value
                img_result[i, j] = 0
    return img_result


def math_bee_pollinator_abundace_v3(
    fa_array: np.ndarray,
    bee_ns_image: np.ndarray,
    alfa: float,
    ratio_x: float,
    ratio_y: float,
    resolution: int,
    multicore: int,
):
    from pollinator_abundance.image_processing import (
        resize_image,
        resize_image_to_target,
    )  # Importing here to avoid circular import issues

    # First, fill NaNs with the mean of surrounding pixels (only inside blobs of non-NaN pixels)
    fa_array = fill_nans_with_neighbors(fa_array)
    bee_ns_image = fill_nans_with_neighbors(bee_ns_image)

    height, width = fa_array.shape
    speed_factor = math.ceil(resolution / ratio_x)
    bee_alfa = alfa / (10 * speed_factor)
    fa_image_resized = resize_image(fa_array, speed_factor)

    bee_fr_image_resized = pixel_mean_calculation_nan_optimized_2D_32bit(
        fa_image_resized, bee_alfa, ratio_x, ratio_y
    )

    bee_fr_image = resize_image_to_target(
        bee_fr_image_resized, width, height
    )  # resize_image(bee_fr_image_resized, 1/speed_factor)

    if bee_ns_image.shape != bee_fr_image.shape:
        bee_ns_image, bee_fr_image = crop_to_same_size(bee_ns_image, bee_fr_image)

    ps_image = bee_ns_image * bee_fr_image

    ps_image_resized = resize_image(ps_image, speed_factor)

    pa_image_step1_resized = pixel_mean_calculation_nan_optimized_2D_32bit(
        ps_image_resized, bee_alfa, ratio_x, ratio_y
    )

    pa_image_step1 = resize_image_to_target(pa_image_step1_resized, width, height)
    pa_image = pa_multiply(pa_image_step1, bee_fr_image, fa_array)

    return (
        np.nanmean(pa_image),
        pa_image,
        bee_ns_image,
        ps_image,
        bee_fr_image,
        speed_factor,
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance
