import numpy as np

from pollinator_abundance.image_processing import (
    apply_mask_to_image,
    create_and_color_image,
    merge_roi_an_ca_array,
)
from pollinator_abundance.math_v2 import image_to_clc_ns_v3
from pollinator_abundance.reporting import create_image_for_reporting


def kpi_elements_generation(
    roi_id,
    ca_id,
    kpi,
    result_values,
    image_all,
    mask_roi,
    mask_ca,
    ref_array,
    palette,
    units,
    speed_factor,
    max_val,
    webp_img,
    webp_report,
    clc_values_roi,
    clc_values_ca,
    input_image_roi,
    input_image_ca,
    alignment_point_x,
    alignment_point_y,
    palette_min,
    palette_max,
    report_palette,
    filename,
    report_ext,
    title_report,
    title_bar,
    width_km_ca,
    height_km_ca,
    width_km_roi,
    height_km_roi,
    bounding_box_roi,
    site_pixel_polygons,
    filename_report,
    value_roi=None,
    value_ca=None,
    min_array_val=0,
    cbar_digits=1,
):
    if kpi != "clc":
        if kpi in ["fa", "msa", "msa_lu_animals", "msa_lu_plants"]:
            array_roi = image_to_clc_ns_v3(input_image_roi, clc_values_roi, kpi)
            array_ca = image_to_clc_ns_v3(input_image_ca, clc_values_ca, kpi)
            ref_array = merge_roi_an_ca_array(
                array_roi, array_ca, alignment_point_x, alignment_point_y
            )

        if value_roi is None:
            roi_vals_masked = np.where(mask_roi, ref_array, np.nan)
            if not np.all(roi_vals_masked != roi_vals_masked):
                value_roi = float(np.nanmean(roi_vals_masked))
            else:
                value_roi = float(np.nanmean(array_roi))
        if value_ca is None:
            ca_vals_masked = np.where(mask_ca, ref_array, np.nan)
            if not np.all(ca_vals_masked != ca_vals_masked):
                value_ca = float(np.nanmean(ca_vals_masked))
            else:
                value_ca = float(np.nanmean(array_ca))

        if kpi in ["pa", "ns"] or any(x in kpi for x in ["ns_", "pa_"]):
            total_mask = mask_roi + mask_ca
            ref_array = np.where(total_mask, ref_array, np.nan)

        image_all = create_and_color_image(
            img_array=ref_array,
            speed_factor=speed_factor,
            max_val=max_val,
            palette_input=palette,
            min_array_val=min_array_val,
        )

        if units is not None and units != "N":
            value_text_ca = f"{title_bar} CA {value_ca:.2f} {units}"
            value_text_roi = (
                f"{title_bar} ROI {value_roi:.2f} {units}"
                if value_roi is not None
                else None
            )

        else:
            value_text_ca = (
                f"{title_bar} CA {value_ca:.2f}" if value_ca is not None else None
            )  # type: ignore[assignment]
            value_text_roi = (
                f"{title_bar} ROI {value_roi:.2f}" if value_roi is not None else None
            )

    else:
        value_roi = None
        value_ca = None
        value_text_ca = None
        value_text_roi = None

    if units is not None:
        final_title_bar = f"{title_bar} [{units}]"
    else:
        final_title_bar = title_bar

    image_ca = apply_mask_to_image(image_all, mask_ca)
    _ = create_image_for_reporting(
        image=image_ca,
        title=f"{title_report} (CA) [ {width_km_ca} x {height_km_ca} km²]",
        x_axis_title="Latitudinal Axis [km]",
        y_axis_title="Longitudinal Axis [km]",
        title_bar=final_title_bar,
        x_axis_scale=[0, width_km_ca],  # type: ignore[arg-type]
        y_axis_scale=[0, height_km_ca],  # type: ignore[arg-type]
        palette_scale=[palette_min, palette_max],  # type: ignore[arg-type]
        palette=report_palette,
        site_pixel_polygon=site_pixel_polygons,
        bounding_box=None,
        value_text=value_text_ca,
        cbar_digits=cbar_digits,
    )

    image_roi = apply_mask_to_image(image_all, mask_roi)
    _ = create_image_for_reporting(
        image=image_roi,
        title=f"{title_report} (ROI) [ {width_km_roi} x {height_km_roi} km²]",
        x_axis_title="Latitudinal Axis [km]",
        y_axis_title="Longitudinal Axis [km]",
        title_bar=final_title_bar,
        x_axis_scale=[0, width_km_roi],  # type: ignore[arg-type]
        y_axis_scale=[0, height_km_roi],  # type: ignore[arg-type]
        palette_scale=[palette_min, palette_max],  # type: ignore[arg-type]
        palette=report_palette,
        site_pixel_polygon=site_pixel_polygons,
        bounding_box=bounding_box_roi,
        value_text=value_text_roi,
        cbar_digits=cbar_digits,
    )

    return None
