import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO

import numpy as np
from importlib.resources import files
import pollinator_abundance
from PIL import Image

from pollinator_abundance.basic import (
    NS_COLUMNS,
    NS_COLUMNS_PA,
    polygons_pixel,
)
from pollinator_abundance.element import kpi_elements_generation
from pollinator_abundance.image_processing import (
    generate_roi_and_ca_mask,
    merge_roi_an_ca_array,
    merge_roi_an_ca_image,
)
from pollinator_abundance.math_v2 import (
    image_to_clc_ns_v3,
    math_bee_pollinator_abundace_v3,
)
from pollinator_abundance.reporting import (
    PALETTE_BLACK_RED_GREEN,
    PALETTE_INPUT,
    PALETTE_PN,
    linear_gradient,
)
from pollinator_abundance.constants import CLC_VALUES, CLC_VALUES_CA, CLC_VALUES_ROI

RESOLUTION_MAP = {
    "super_res": (25, 0),
    "high": (50, 0),
    "low": (100, 0),
    "fast": (500, 0),
    "big": (100, 32),
}

DATA_BEE_STR = """
    SPECIES\tns\tns_soilexcavators\tns_sandexcavators\tns_underground_cavities\tns_aboveground_cavities_wetland\tns_aboveground_cavities_vegetated\tns_coastal_area\tforaging_activity_allseasons_index\talpha\trelative_abundance\toccurrences\tITD\t# flight months\tMonth Start\tMonth end\tns_artificial
    Osmia bicornis\t1\t0\t0\t0\t0\t1\t0\t0.33\t2257.58\t1\t46301\t2.98\t4\t4\t7\t0
    Ceratina cucurbitina\t1\t0\t0\t0\t0\t1\t0\t0.50\t1060.61\t1\t2417\t1.4\t6\t4\t9\t0
    Anthidium manicatus\t1\t0\t0\t0\t0\t1\t0\t0.25\t2727.27\t1\t18830\t3.6\t3\t6\t9\t0
    Hylaeus gibbus\t1\t0\t0\t0\t0\t1\t0\t0.25\t909.09\t1\t1412\t1.2\t3\t6\t8\t0
    Megachile pilidens\t1\t0\t0\t0\t0\t1\t0\t0.33\t2007.58\t1\t1864\t2.65\t4\t6\t9\t0
    Bombus terrestris\t1\t0\t0\t1\t0\t0\t0\t0.67\t4477.27\t1\t207062\t5.91\t8\t3\t10\t0
    Colletes cunicularius\t1\t0\t1\t0\t0\t0\t0\t0.25\t2560.61\t1\t17594\t3.38\t3\t4\t6\t0
    Dasypoda hirtipes\t1\t0\t1\t0\t0\t0\t0\t0.33\t2068.18\t1\t25210\t2.73\t4\t6\t9\t0
    Amegilla quadrifasciata\t1\t0\t0\t0\t0\t0\t1\t0.25\t2916.67\t1\t480\t3.85\t3\t6\t8\t0
    Andrena flavipes\t1\t1\t0\t0\t0\t0\t0\t0.25\t2113.64\t1\t28231\t2.79\t3\t7\t9\t0
    Lasioglossum malachurum\t1\t1\t0\t0\t0\t0\t0\t0.58\t1212.12\t1\t12503\t1.6\t7\t4\t10\t0
    Halictus scabiosae\t1\t1\t0\t0\t0\t0\t0\t0.50\t1916.67\t1\t14830\t2.53\t6\t4\t9\t0
    Hylaeus hyalinatus\t1\t0\t0\t0\t1\t0\t0\t0.33\t909.09\t1\t9283\t1.2\t4\t5\t9\t0
    Apis Mellifera\t1\t0\t0\t0\t0\t0\t0\t0.33\t3300.09\t1\t9283\t1.2\t4\t5\t9\t1
    """.strip()


def parse_lambda_event(event):
    """
    This function parses to JSON the 'body' key of the 'event' object.
    If 'event' is missing that key, this function returns the 'event' object itself.
    """
    if "body" in event:
        return json.loads(event.get("body"))
    return event


def pa_single_bee_roi_ca(event, context):
    clc_values_roi = event["clc_values_roi"]
    clc_values_ca = event["clc_values_ca"]
    alignment_point_x = event["alignment_point_x"]
    alignment_point_y = event["alignment_point_y"]
    ratio_x = event["ratio_x"]
    ratio_y = event["ratio_y"]
    multicore = event.get("multicore", 0)
    bee = event["bee"]
    resolution = event.get("resolution", 200)
    ns_columns = event.get("ns_columns", NS_COLUMNS)

    # Read from data
    path_to_image_roi_np = files(pollinator_abundance) / "data/image_roi.npy"
    image_roi_np = np.load(path_to_image_roi_np)
    image_roi = Image.fromarray(image_roi_np)

    path_to_image_ca_np = files(pollinator_abundance) / "data/image_ca.npy"
    image_ca_np = np.load(path_to_image_ca_np)
    image_ca = Image.fromarray(image_ca_np)

    ns_bee = next((col for col in ns_columns if float(bee[col]) == 1), None)

    array_fa_roi = image_to_clc_ns_v3(image_roi, clc_values_roi, "fa")
    array_fa_ca = image_to_clc_ns_v3(image_ca, clc_values_ca, "fa")
    array_fa = merge_roi_an_ca_array(
        array_fa_roi, array_fa_ca, alignment_point_x, alignment_point_y
    )

    array_ns_bee_roi = image_to_clc_ns_v3(image_roi, clc_values_roi, ns_bee)
    array_ns_bee_ca = image_to_clc_ns_v3(image_ca, clc_values_ca, ns_bee)
    array_ns_bee = merge_roi_an_ca_array(
        array_ns_bee_roi, array_ns_bee_ca, alignment_point_x, alignment_point_y
    )

    # print()
    # print(f"bee {bee['SPECIES']}, array_fa.shape: ", array_fa.shape)
    # print(f"bee {bee['SPECIES']}, array_fa.dtype: ", array_fa.dtype)
    # print(f"bee {bee['SPECIES']}, array_fa.max: ", np.nanmax(array_fa))
    # print(f"bee {bee['SPECIES']}, array_fa.min: ", np.nanmin(array_fa))
    # print(f"bee {bee['SPECIES']}, array_fa nan: ", np.isnan(array_fa).sum())
    # print()

    # print()
    # print(f"bee {bee['SPECIES']}, array_ns_bee.shape: ", array_ns_bee.shape)
    # print(f"bee {bee['SPECIES']}, array_ns_bee.dtype: ", array_ns_bee.dtype)
    # print(f"bee {bee['SPECIES']}, array_ns_bee.max: ", np.nanmax(array_ns_bee))
    # print(f"bee {bee['SPECIES']}, array_ns_bee.min: ", np.nanmin(array_ns_bee))
    # print(f"bee {bee['SPECIES']}, array_ns_bee nan: ", np.isnan(array_ns_bee).sum())
    # print()

    pa_value, pa_image, ns_image, ps_image, bee_fr_image, speed_factor = (
        math_bee_pollinator_abundace_v3(
            array_fa,
            array_ns_bee,
            float(bee["alpha"]),
            ratio_x,
            ratio_y,
            resolution,
            multicore,
        )
    )

    # print()
    # print(f"bee {bee['SPECIES']}, pa_image.shape: ", pa_image.shape)
    # print(f"bee {bee['SPECIES']}, pa_image.dtype: ", pa_image.dtype)
    # print(f"bee {bee['SPECIES']}, pa_image.max: ", np.nanmax(pa_image))
    # print(f"bee {bee['SPECIES']}, pa_image.min: ", np.nanmin(pa_image))
    # print(f"bee {bee['SPECIES']}, pa_image nan: ", np.isnan(pa_image).sum())
    # print()

    return pa_value, pa_image, ns_image, ps_image


def lambda_bee(
    plantation_id,
    bee,
    clc_values_roi,
    clc_values_ca,
    roi,
    ca,
    ratio_x,
    ratio_y,
    min_res,
    image_url_fa,
    ns_columns=NS_COLUMNS,
    multicore=0,
    plantations_polygons_id=0,
    override=True,
    how="lambda",
):
    print(f"Performing lambda_bee for bee {bee['SPECIES']}")
    lambda_payload = {
        "plantation_id": plantation_id,
        "plantations_polygons_id": plantations_polygons_id,
        "clc_layer_id_roi": roi["id"],
        "clc_layer_id_ca": ca["id"],
        "clc_values_roi": clc_values_roi,
        "clc_values_ca": clc_values_ca,
        "image_url_roi": roi["image_url"],
        "image_url_ca": ca["image_url"],
        "alignment_point_x": roi["alignment_point_x"],
        "alignment_point_y": roi["alignment_point_y"],
        "ratio_x": ratio_x,
        "ratio_y": ratio_y,
        "bee": bee,
        "resolution": min_res,
        "ns_columns": ns_columns,
        "multicore": multicore,
        "override": override,
        "image_url_fa": image_url_fa,
    }

    pa_value, pa_image, ns_image, ps_image = pa_single_bee_roi_ca(lambda_payload, {})

    ns_name = next((ns_col for ns_col in ns_columns if bee.get(ns_col) == "1"), None)

    return ns_name, pa_image, ns_image


def pollinator_abundance_calculation():
    """Main function to calculate the Pollinator Abundance (PA) and Nectar Potential (NP) for a given plantation and ROI."""
    start_lt = time.time()

    dict_of_results = {}

    # Hardcoded inputs
    plantation_id = 9827
    plantations_polygons_id = 9773
    resolution = "low"
    ca_id = 284085
    roi_id = 284086
    override_bee = True
    how = "local"
    compute_pa_ns = True
    compute_only_msa = False

    min_res, multicore = RESOLUTION_MAP.get(resolution, (200, 0))

    print(f"Got plantation_id: {plantation_id}, roi_id: {roi_id}, ca_id: {ca_id}")

    # Set ratio
    ratio_x = 5.674733628978614
    ratio_y = 5.662378135559605

    # Set ROI and CA indices
    roi = {
        "plantation_id": 9827,
        "id": 284086,
        "alignment_point_x": 198.0,
        "alignment_point_y": 289.0,
        "ratio_x": 5.674733628978614,
        "ratio_y": 5.662378135559605,
        "bbox": '{"type": "Polygon", "coordinates": [[[9.09288174073389, 45.80396180948701], [9.09288174073389, 45.82143582246963], [9.058201466780092, 45.82143582246963], [9.058201466780092, 45.80396180948701], [9.09288174073389, 45.80396180948701]]]}',
        "plantations_polygons_id": 9773,
        "related_at": datetime(2024, 7, 1, 0, 0),
        "image_url": None,
    }
    ca = {
        "plantation_id": 9827,
        "id": 284085,
        "alignment_point_x": 0.0,
        "alignment_point_y": 0.0,
        "ratio_x": 5.674644900406369,
        "ratio_y": 5.662291983909758,
        "bbox": '{"type": "Polygon", "coordinates": [[[9.108287083479684, 45.790967372071584], [9.108287083479684, 45.836154568894095], [9.043745268374089, 45.836154568894095], [9.043745268374089, 45.790967372071584], [9.108287083479684, 45.790967372071584]]]}',
        "plantations_polygons_id": 9773,
        "related_at": datetime(2024, 7, 1, 0, 0),
        "image_url": None,
    }

    try:
        dict_of_results["ratio_x"] = ratio_x
        dict_of_results["ratio_y"] = ratio_y

        # Initialize Result Values dict
        result_values = {
            "CA": {
                "PA": None,
                "FA": None,
                "NP": None,
                "NS": None,
                "MSA": None,
                "MSA_LU_ANIMALS": None,
                "MSA_LU_PLANTS": None,
            },
            "ROI": {
                "PA": None,
                "FA": None,
                "NP": None,
                "NS": None,
                "MSA": None,
                "MSA_LU_ANIMALS": None,
                "MSA_LU_PLANTS": None,
            },
            "Delta": {
                "PA": None,
                "FA": None,
                "NP": None,
                "NS": None,
                "MSA": None,
                "MSA_LU_ANIMALS": None,
                "MSA_LU_PLANTS": None,
            },
        }

        # Get ROI and CA Images from saved data
        path_to_np_image_roi = files(pollinator_abundance) / "data/np_image_roi.npy"
        np_image_roi = Image.fromarray(np.load(path_to_np_image_roi))

        path_to_np_image_ca = files(pollinator_abundance) / "data/np_image_ca.npy"
        np_image_ca = Image.fromarray(np.load(path_to_np_image_ca))

        width_km_ca = 5.0
        height_km_ca = 5.0

        alignment_point_x = 198.0
        alignment_point_y = 289.0

        dict_of_results["np_image_roi"] = np_image_roi
        dict_of_results["np_image_ca"] = np_image_ca
        dict_of_results["width_km_ca"] = width_km_ca
        dict_of_results["height_km_ca"] = height_km_ca
        dict_of_results["alignment_point_x"] = alignment_point_x
        dict_of_results["alignment_point_y"] = alignment_point_y

        print("Got images and dimensions")

        # Retrieve Site pixel polygons and ROI's bbox
        site_pixel_polygons, bounding_box_roi = polygons_pixel(ca["id"])
        width_km_roi, height_km_roi = (
            round(((bounding_box_roi[2] - bounding_box_roi[0]) * ratio_x / 1000), 1),
            round(((bounding_box_roi[3] - bounding_box_roi[1]) * ratio_y / 1000), 1),
        )

        # Merge ROI and CA images
        site_pixel_polygons = [
            np.array(polygon, dtype=np.int32) for polygon in site_pixel_polygons
        ]
        image_all = merge_roi_an_ca_image(
            np_image_roi, np_image_ca, alignment_point_x, alignment_point_y
        )

        dict_of_results["site_pixel_polygons"] = site_pixel_polygons
        dict_of_results["bounding_box_roi"] = bounding_box_roi
        dict_of_results["image_all"] = image_all

        print("Merged ROI and CA images")

        # Get CLC values
        clc_values = CLC_VALUES
        clc_values_roi = CLC_VALUES_CA
        clc_values_ca = CLC_VALUES_ROI

        dict_of_results["clc_values"] = clc_values
        dict_of_results["clc_values_roi"] = clc_values_roi
        dict_of_results["clc_values_roi"] = clc_values_roi

        print(
            f"Setting folder plantation_id/plantations_polygons_id as: {plantation_id}/{plantations_polygons_id}"
        )

        array_pn_roi = image_to_clc_ns_v3(np_image_roi, clc_values_roi, "pn_mean")
        array_pn_ca = image_to_clc_ns_v3(np_image_ca, clc_values_ca, "pn_mean")

        array_pn = merge_roi_an_ca_array(
            array_pn_roi, array_pn_ca, alignment_point_x, alignment_point_y
        )

        dict_of_results["array_pn_roi"] = array_pn_roi
        dict_of_results["array_pn_ca"] = array_pn_ca
        dict_of_results["array_pn"] = array_pn

        mex = "Retrieved CLC data from DB"
        print(mex)

        mask_roi_field, mask_ca = generate_roi_and_ca_mask(
            array_pn=array_pn,
            site_pixel_polygons=site_pixel_polygons,
        )

        dict_of_results["mask_roi_field"] = mask_roi_field
        dict_of_results["mask_ca"] = mask_ca

        if not compute_only_msa:
            try:
                ### CLC
                kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi="clc",
                    result_values=None,
                    image_all=image_all,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=None,
                    palette=None,
                    report_palette=None,
                    units="",
                    palette_min=0,
                    palette_max=100,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    speed_factor=None,
                    max_val=None,
                    webp_img=True,
                    webp_report=True,
                    filename="clc",
                    title_report="Corine Land Cover",
                    title_bar="CLC",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report="clc_report",
                    report_ext="",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )
            except Exception as e:
                raise e

            mex = "Created CLC images"
            print(mex)

            ### NECTAR POTENTIAL

            try:
                kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi="np",
                    result_values=result_values,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=array_pn,
                    palette=PALETTE_PN,
                    report_palette=linear_gradient(PALETTE_PN, n=256)[::-1],
                    units="kg/ha/year",
                    palette_min=0,
                    palette_max=250,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    speed_factor=1,
                    max_val=2,
                    webp_img=False,
                    webp_report=True,
                    filename="np",
                    title_report="Nectariferous Potential (NP)",
                    title_bar="NP",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report="pn_report",
                    report_ext="",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )
            except Exception as e:
                raise e

            mex = "Created PN images"
            print(mex)

            ### FLOWER AVAILABILITY

            try:
                image_url_fa = kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi="fa",
                    result_values=result_values,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=None,
                    report_palette=linear_gradient(PALETTE_INPUT, n=256)[::-1],
                    units="N",
                    palette_min=0,
                    palette_max=1,
                    clc_values_roi=clc_values_roi,
                    clc_values_ca=clc_values_ca,
                    speed_factor=1,
                    max_val=255,
                    palette=PALETTE_INPUT,
                    webp_img=False,
                    webp_report=True,
                    input_image_roi=np_image_roi,
                    input_image_ca=np_image_ca,
                    alignment_point_x=alignment_point_x,
                    alignment_point_y=alignment_point_y,
                    filename="fa",
                    title_report="Pollinator Foraging Activity (FA)",
                    title_bar="FA",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report="fa_report",
                    report_ext=".webp",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )
            except Exception as e:
                raise e

            mex = "Created FA images"
            print(mex)

        ### MSA (LU, all taxonomic groups)

        try:
            kpi_elements_generation(
                roi_id=roi["id"],
                ca_id=ca["id"],
                kpi="msa",
                result_values=result_values,
                image_all=None,
                mask_roi=mask_roi_field,
                mask_ca=mask_ca,
                ref_array=None,
                report_palette=linear_gradient(PALETTE_BLACK_RED_GREEN, n=256)[::-1],
                units="N",
                palette_min=0,
                palette_max=1,
                clc_values_roi=clc_values_roi,
                clc_values_ca=clc_values_ca,
                speed_factor=1,
                max_val=255,
                palette=PALETTE_BLACK_RED_GREEN,
                webp_img=True,
                webp_report=True,
                input_image_roi=np_image_roi,
                input_image_ca=np_image_ca,
                alignment_point_x=alignment_point_x,
                alignment_point_y=alignment_point_y,
                filename="msa",
                title_report="Mean Species Abundance (MSA)",
                title_bar="MSA",
                width_km_ca=width_km_ca,
                height_km_ca=height_km_ca,
                width_km_roi=width_km_roi,
                height_km_roi=height_km_roi,
                bounding_box_roi=bounding_box_roi,
                site_pixel_polygons=site_pixel_polygons,
                filename_report="msa_report",
                report_ext=".webp",
                value_roi=None,
                value_ca=None,
                min_array_val=0,
                cbar_digits=1,
            )
        except Exception as e:
            raise e
        mex = "Created MSA images"
        print(mex)

        ### MSA_LU_animals
        try:
            kpi_elements_generation(
                roi_id=roi["id"],
                ca_id=ca["id"],
                kpi="msa_lu_animals",
                result_values=result_values,
                image_all=None,
                mask_roi=mask_roi_field,
                mask_ca=mask_ca,
                ref_array=None,
                report_palette=linear_gradient(PALETTE_BLACK_RED_GREEN, n=256)[::-1],
                units="N",
                palette_min=0,
                palette_max=1,
                clc_values_roi=clc_values_roi,
                clc_values_ca=clc_values_ca,
                speed_factor=1,
                max_val=255,
                palette=PALETTE_BLACK_RED_GREEN,
                webp_img=True,
                webp_report=True,
                input_image_roi=np_image_roi,
                input_image_ca=np_image_ca,
                alignment_point_x=alignment_point_x,
                alignment_point_y=alignment_point_y,
                filename="msa_lu_animals",
                title_report="Mean Species Abundance for Land Use (MSA_LU) - Animals",
                title_bar="MSA_LU",
                width_km_ca=width_km_ca,
                height_km_ca=height_km_ca,
                width_km_roi=width_km_roi,
                height_km_roi=height_km_roi,
                bounding_box_roi=bounding_box_roi,
                site_pixel_polygons=site_pixel_polygons,
                filename_report="msa_lu_animals_report",
                report_ext=".webp",
                value_roi=None,
                value_ca=None,
                min_array_val=0,
                cbar_digits=1,
            )
        except Exception as e:
            raise e
        mex = "Created MSA_LU Animals images"
        print(mex)

        ### MSA_LU_plants

        try:
            kpi_elements_generation(
                roi_id=roi["id"],
                ca_id=ca["id"],
                kpi="msa_lu_plants",
                result_values=result_values,
                image_all=None,
                mask_roi=mask_roi_field,
                mask_ca=mask_ca,
                ref_array=None,
                report_palette=linear_gradient(PALETTE_BLACK_RED_GREEN, n=256)[::-1],
                units="N",
                palette_min=0,
                palette_max=1,
                clc_values_roi=clc_values_roi,
                clc_values_ca=clc_values_ca,
                speed_factor=1,
                max_val=255,
                palette=PALETTE_BLACK_RED_GREEN,
                webp_img=True,
                webp_report=True,
                input_image_roi=np_image_roi,
                input_image_ca=np_image_ca,
                alignment_point_x=alignment_point_x,
                alignment_point_y=alignment_point_y,
                filename="msa_lu_plants",
                title_report="Mean Species Abundance for Land Use (MSA_LU) - Plants",
                title_bar="MSA_LU",
                width_km_ca=width_km_ca,
                height_km_ca=height_km_ca,
                width_km_roi=width_km_roi,
                height_km_roi=height_km_roi,
                bounding_box_roi=bounding_box_roi,
                site_pixel_polygons=site_pixel_polygons,
                filename_report="msa_lu_plants_report",
                report_ext=".webp",
                value_roi=None,
                value_ca=None,
                min_array_val=0,
                cbar_digits=1,
            )
        except Exception as e:
            raise e
        mex = "Created MSA_LU Plants images"
        print(mex)

        # According to parameter 'compute_pa_ns', compute or skip PA and NS
        if compute_pa_ns is True:
            data_io = StringIO(DATA_BEE_STR)
            bee_data = csv.DictReader(data_io, delimiter="\t")
            bee_data = [x for x in bee_data]  # type: ignore[assignment]
            ns_sum_roi = {ns_col: 0.0 for ns_col in NS_COLUMNS}
            ns_sum_ca = {ns_col: 0.0 for ns_col in NS_COLUMNS}

            pa_bees_image_ns = {ns_col: None for ns_col in NS_COLUMNS}
            ns_images = {ns_col: None for ns_col in NS_COLUMNS}
            total_ns_count = {ns_col: 0 for ns_col in NS_COLUMNS}
            max_threads = 2
            total_bee = 0

            dict_of_results["bee_data"] = bee_data

            print("Running ThreadPool")

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [
                    executor.submit(
                        lambda_bee,
                        plantation_id,
                        bee,
                        clc_values_roi,
                        clc_values_ca,
                        roi,
                        ca,
                        ratio_x,
                        ratio_y,
                        min_res,
                        image_url_fa,
                        NS_COLUMNS,
                        multicore,
                        plantations_polygons_id,
                        override_bee,
                        how,
                    )
                    for bee in bee_data
                ]
                for future in futures:
                    ns_name, pa_image, ns_image = future.result()
                    if pa_image is not None:
                        if pa_bees_image_ns[ns_name] is None:
                            pa_bees_image_ns[ns_name] = np.zeros_like(pa_image)

                        if ns_images[ns_name] is None:
                            ns_images[ns_name] = np.zeros_like(ns_image)
                        ns_images[ns_name] += ns_image
                        ns_sum_roi[ns_name] += np.nanmean(
                            np.where(mask_roi_field, ns_image, np.nan)
                        )
                        ns_sum_ca[ns_name] += np.nanmean(
                            np.where(mask_ca, ns_image, np.nan)
                        )
                        pa_bees_image_ns[ns_name] += pa_image
                        total_ns_count[ns_name] += 1
                        total_bee += 1

            mex = "Computed PA and NS data"
            print(mex)

            pa_image_total = np.zeros_like(pa_image)
            ns_images_total = np.zeros_like(ns_image)

            dict_of_results["pa_image_total"] = pa_image_total
            dict_of_results["ns_images_total"] = ns_images_total

            i = 0
            artificial_bee = False
            total_ns_pa_cycle = len(NS_COLUMNS)
            for idx, ns in enumerate(NS_COLUMNS):
                if total_ns_count[ns] != 0:
                    pa_image_total += pa_bees_image_ns[ns]
                    ns_images_total += ns_images[ns]

                pa_bee_image_n_normalized = pa_bees_image_ns[ns] / total_ns_count[ns]  # type: ignore[operator]

                dict_of_results[f"pa_bee_image_n_normalized_{idx}"] = (
                    pa_bee_image_n_normalized
                )

                _ = kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi=f"pa_{ns}",
                    result_values=None,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=pa_bee_image_n_normalized,
                    report_palette=linear_gradient(PALETTE_INPUT, n=256)[::-1],
                    units="N",
                    palette_min=0,
                    palette_max=0.4,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    speed_factor=1,
                    max_val=255 * 2.5,
                    palette=PALETTE_INPUT,
                    webp_img=True,
                    webp_report=True,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    filename=f"pa_{ns}.png",
                    title_report=f"{NS_COLUMNS_PA[i]} (PA)",
                    title_bar="PA",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report=f"{ns}_pa_report",
                    report_ext=".webp",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )

                ns_images_n_normalized = ns_images[ns] / total_ns_count[ns]  # type: ignore[operator]

                dict_of_results[f"ns_images_n_normalized_{idx}"] = (
                    ns_images_n_normalized
                )

                kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi=f"ns_{ns}",
                    result_values=None,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=ns_images_n_normalized,
                    report_palette=linear_gradient(PALETTE_INPUT, n=256)[::-1],
                    units="N",
                    palette_min=0,
                    palette_max=0.5,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    speed_factor=1,
                    max_val=255 * 2,
                    palette=PALETTE_INPUT,
                    webp_img=True,
                    webp_report=True,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    filename=f"ns_{ns}.png",
                    title_report=f"{NS_COLUMNS_PA[i]} (NS)",
                    title_bar="NS",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report=f"{ns}_ns_report",
                    report_ext=".webp",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )

                i += 1
                mex = f"Creating NS and PA: step {idx + 1}/{total_ns_pa_cycle}"
                print(mex)
            print("Created images PA and NS")

            if not artificial_bee:
                total_bee = total_bee - 1

            pa_image_total_normalized = pa_image_total / total_bee

            dict_of_results["pa_image_total_normalized"] = pa_image_total_normalized

            try:
                kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi="pa",
                    result_values=result_values,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=pa_image_total_normalized,
                    report_palette=linear_gradient(PALETTE_INPUT, n=256)[::-1],
                    units="N",
                    palette_min=0,
                    palette_max=0.4,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    speed_factor=1,
                    max_val=255 * 2.5,
                    palette=PALETTE_INPUT,
                    webp_img=True,
                    webp_report=True,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    filename="PA_TOTAL.png",
                    title_report="Pollinator Abundance (PA)",
                    title_bar="PA",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report="pa_report",
                    report_ext=".webp",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )
            except Exception as e:
                raise e

            mex = "Created PA images"
            print(mex)

            ns_image_total_normalized = ns_images_total / total_bee

            dict_of_results["ns_image_total_normalized"] = ns_image_total_normalized

            try:
                kpi_elements_generation(
                    roi_id=roi["id"],
                    ca_id=ca["id"],
                    kpi="ns",
                    result_values=result_values,
                    image_all=None,
                    mask_roi=mask_roi_field,
                    mask_ca=mask_ca,
                    ref_array=ns_image_total_normalized,
                    report_palette=linear_gradient(PALETTE_INPUT, n=256)[::-1],
                    units="N",
                    palette_min=0,
                    palette_max=0.5,
                    clc_values_roi=None,
                    clc_values_ca=None,
                    speed_factor=1,
                    max_val=255 * 2.5,
                    palette=PALETTE_INPUT,
                    webp_img=True,
                    webp_report=True,
                    input_image_roi=None,
                    input_image_ca=None,
                    alignment_point_x=None,
                    alignment_point_y=None,
                    filename="ns_total.png",
                    title_report="Nesting Suitability (NS)",
                    title_bar="NS",
                    width_km_ca=width_km_ca,
                    height_km_ca=height_km_ca,
                    width_km_roi=width_km_roi,
                    height_km_roi=height_km_roi,
                    bounding_box_roi=bounding_box_roi,
                    site_pixel_polygons=site_pixel_polygons,
                    filename_report="ns_report",
                    report_ext=".webp",
                    value_roi=None,
                    value_ca=None,
                    min_array_val=0,
                    cbar_digits=1,
                )
            except Exception as e:
                raise e

            mex = "Created NS images"
            print(mex)

        dict_of_results["result_values"] = result_values

    except Exception as e:
        print(f"pa_integrated_fast_v2 - Exception: {e}")
        raise e

    # Task completed!
    print("Task completed!")
    print(f"\n\nelapsed total {time.time() - start_lt} s")
    return dict_of_results
