import numpy as np


def crop_to_same_size(img1, img2):
    # Calcola le dimensioni minime comuni
    # shape[0] è l'altezza, shape[1] è la larghezza
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    # Funzione per croppare un'immagine centrata sulle dimensioni minime
    def crop_center(img):
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
        half_height, half_width = min_height // 2, min_width // 2
        return img[
            center_y - half_height : center_y + half_height,
            center_x - half_width : center_x + half_width,
        ]

    cropped_img1 = crop_center(img1)
    cropped_img2 = crop_center(img2)

    return cropped_img1, cropped_img2


def hex_to_fa(hex_color, hex_to_fa_table):
    # Cerca il valore fa corrispondente al colore hex
    for record in hex_to_fa_table:
        if record["color"].lower() == hex_color.lower():
            return record["fa"]
    return None


def image_hex_to_fa(image, hex_to_fa_list):
    hex_to_fa_table = {item["color"].lower(): item["fa"] for item in hex_to_fa_list}
    img_array = np.array(image)
    altezza, larghezza, _ = img_array.shape
    fa_array = np.full((altezza, larghezza), np.nan)
    hex_array = np.apply_along_axis(
        lambda row: "#" + "".join([f"{val:02x}" for val in row[:3]]), 2, img_array
    )

    for hex_color, fa_value in hex_to_fa_table.items():
        mask = hex_array == hex_color
        fa_array[mask] = fa_value

    return fa_array


def pixel_mean_calculation_nan(img_array, alfa, ratio_x, ratio_y):
    altezza, larghezza = img_array.shape
    # Inizializza img_result con NaN
    img_result = np.full((altezza, larghezza), np.nan)

    # Creazione delle coordinate dei pixel
    x_coords, y_coords = np.meshgrid(np.arange(larghezza), np.arange(altezza))
    x_coords = x_coords * ratio_x
    y_coords = y_coords * ratio_y

    # Crea una maschera per identificare dove si trovano i NaN
    nan_mask = np.isnan(img_array)

    for i in range(altezza):
        for j in range(larghezza):
            # Salta il calcolo per i pixel già NaN
            if nan_mask[i, j]:
                continue

            # Calcolo delle distanze in modo vettorializzato
            distanze_x = (i - y_coords) ** 2
            distanze_y = (j - x_coords) ** 2
            distanze = np.sqrt(distanze_x + distanze_y)

            # Calcolo dei pesi esponenziali
            pesi = np.exp(-distanze / alfa)

            # Applica la maschera per ignorare i NaN nella somma pesata e nel peso totale
            pesi_masked = np.where(nan_mask, 0, pesi)
            img_array_masked = np.where(nan_mask, 0, img_array)

            somma_pesata = np.sum(img_array_masked * pesi_masked)
            peso_totale = np.sum(pesi_masked)

            # Gestisci il caso in cui tutti i pesi siano zero per evitare la divisione per zero
            if peso_totale != 0:
                img_result[i, j] = somma_pesata / peso_totale

    return img_result


def pa_multiply(pa_image_step1, fr_image, fa_image):
    # Replace zeros in fr_image with NaN to avoid division by zero
    fr_image = np.where(fr_image == 0, np.nan, fr_image)
    img_result = pa_image_step1 * fa_image / fr_image
    return img_result
