import json
import logging
import math

import numpy as np

from pollinator_abundance.image_processing import find_bounding_box
from pollinator_abundance.logconf import create_logger

logger = create_logger(name=__name__, level=logging.INFO)

NS_COLUMNS = [
    "ns_soilexcavators",
    "ns_sandexcavators",
    "ns_underground_cavities",
    "ns_aboveground_cavities_wetland",
    "ns_aboveground_cavities_vegetated",
    "ns_coastal_area",
    "ns_artificial",
]

NS_COLUMNS_PA = [
    "Soil Excavators",
    "Sand Excavators",
    "Underground Cavities",
    "Aboveground Cavities Wetland",
    "Aboveground Cavities Vegetated",
    "Coastal Area",
    "Artificiale",
]

ALL_CLC_KEY = NS_COLUMNS + [
    "msa",
    "fa",
    "ns",
    "hectare",
    "pn_mean",
    "msa_cc",
    "msa_lu_animals",
    "msa_lu_plants",
]


def haversine(lat1, lon1, lat2, lon2):
    # Raggio della Terra in metri
    R = 6371000
    # Conversione da gradi a radianti
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def parse_lambda_event(event):
    """
    This function parses to JSON the 'body' key of the 'event' object.
    If 'event' is missing that key, this function returns the 'event' object itself.
    """
    if "body" in event:
        return json.loads(event.get("body"))
    return event


def merge_dicts(dicts):
    """Function to merge a list of dictionaries, concatenating values with a space"""
    final_dict = {}  # type: ignore[var-annotated]
    # Iterate over each dictionary in the list
    for d in dicts:
        for key, value in d.items():
            if key in final_dict:
                final_dict[key] += " " + value  # Concatenate values with a space
            else:
                final_dict[key] = (
                    value  # Assign value if key is not present in final_dict
                )
    return final_dict


def average_and_combine_by_color(clc_values):
    """
    This function groups the elements in 'clc_values' by color and computes the mean value
    of each parameter across the same color.
    """

    # Group values by color
    grouped_by_color = {}  # type: ignore[var-annotated]
    for row in clc_values:
        color = row.get("color")
        if color in grouped_by_color:
            grouped_by_color[color].append(row)
        else:
            grouped_by_color[color] = [row]

    # Compute mean of values for each color group
    averaged_values = []
    for color, rows in grouped_by_color.items():
        avg_row = {}
        names = [row.get("name") for row in rows]
        names_i18n = [row.get("name_i18n") for row in rows]
        for row in rows:
            for key, value in row.items():
                if key == "color":
                    avg_row[key] = value  # Color is constant
                elif key in ALL_CLC_KEY:
                    avg_row[key] = (
                        avg_row.get(key, 0) + value
                        if value is not None
                        else avg_row.get(key)
                    )

        # Compute mean for numeric columns
        for key in avg_row:
            if key not in ["color", "name", "name_i18n"] and avg_row[key] is not None:
                avg_row[key] /= len(rows)

        # Create new name if there are more than one, otherwise use original name
        avg_row["name"] = (
            " ".join(set(name for name in names)) if len(names) > 1 else names[0]
        )
        if len(names_i18n) == 1:
            avg_row["name_i18n"] = names_i18n[0]
        else:
            names_i18n_dicts = [json.loads(x) for x in names_i18n]
            name_i18n_dict = merge_dicts(names_i18n_dicts)
            avg_row["name_i18n"] = json.dumps(name_i18n_dict)
        averaged_values.append(avg_row)

    return averaged_values


def scale_polygons_and_bbox(polygons, bbox, scale_factor):
    """
    Scala i poligoni e il bounding box di un fattore dato.

    Args:
    - polygons: Lista di poligoni (ogni poligono è una lista di tuple (x, y))
    - bbox: Bounding box in formato (left, upper, right, lower)
    - scale_factor: Fattore di scala (float o int)

    Returns:
    - scaled_polygons: Lista di poligoni scalati
    - scaled_bbox: Bounding box scalato
    """
    # Scala i poligoni
    scaled_polygons = []
    for polygon in polygons:
        scaled_polygon = [(x * scale_factor, y * scale_factor) for x, y in polygon]
        scaled_polygons.append(scaled_polygon)

    # Scala il bounding box
    left, upper, right, lower = bbox
    scaled_bbox = (
        left * scale_factor,
        upper * scale_factor,
        right * scale_factor,
        lower * scale_factor,
    )

    return scaled_polygons, scaled_bbox


def polygons_pixel(clc_roi_id):
    """
    This function retrieves the 'polygons_pixel' field for the given CLC Layer and returns it together
    with the associated bounding box
    """
    # HARD-CODED POLYGON
    polygons_pixel = [[[198, 289], [673, 306], [659, 632], [203, 557], [198, 289]]]
    all_points = [point for polygon in polygons_pixel for point in polygon]
    return polygons_pixel, find_bounding_box(all_points, 5)


def get_site_pixel_polygons_bounding_box_width_height(ca, ratio_x, ratio_y):
    # Get Polygon, bounding box, ROI width and height
    site_pixel_polygons, bounding_box_roi = polygons_pixel(ca["id"])
    width_km_roi, height_km_roi = (
        round(((bounding_box_roi[2] - bounding_box_roi[0]) * ratio_x / 1000), 1),
        round(((bounding_box_roi[3] - bounding_box_roi[1]) * ratio_y / 1000), 1),
    )
    site_pixel_polygons = [
        np.array(polygon, dtype=np.int32) for polygon in site_pixel_polygons
    ]
    return site_pixel_polygons, bounding_box_roi, width_km_roi, height_km_roi
