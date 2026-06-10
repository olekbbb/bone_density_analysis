import glob
import os

import napari
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def stack(batch_name):
    im_sample = Image.open(batch_name[0])
    orig_w, orig_h = im_sample.size

    imgs = np.zeros((len(batch_name), orig_h, orig_w), dtype=np.float32)

    for i, imfile in enumerate(tqdm(batch_name, desc="Wczytywanie obrazow")):
        with Image.open(imfile) as inputslice:
            imgs[i, :, :] = np.array(inputslice)

    return imgs


def view(stack, mask_kora, mask_bryla):
    viewer = napari.Viewer()
    viewer.add_image(stack, name="Oryginal")
    viewer.add_labels(mask_kora, name="Tkanka Zbita")
    viewer.add_labels(mask_bryla, name="Pelna Bryla", opacity=0.3)
    napari.run()


def calculate_mechanics(volume, mask_kora, mask_bryla, pixel_size=0.5):
    z, h, w = volume.shape
    results = []

    # Params from article.
    a, b = 0.000785524, 0.004277819
    c, d = 0.079, 0.877
    e, f = 3.891, 2.39

    pixel_min, pixel_max = np.min(volume), np.max(volume)
    hu_min, hu_max = -600, 1500

    def scale_to_hu(pixel_val):
        return hu_min + (pixel_val - pixel_min) * (hu_max - hu_min) / (pixel_max - pixel_min)

    for i in tqdm(range(z), desc="Analiza przekrojow"):
        slice_gray = volume[i]
        mask_cortex = mask_kora[i] > 0
        mask_total = mask_bryla[i] > 0

        if not np.any(mask_cortex):
            continue

        area_cortex = np.sum(mask_cortex) * (pixel_size**2)
        area_total = np.sum(mask_total) * (pixel_size**2)
        cortical_ratio = area_cortex / area_total if area_total > 0 else 0

        # Konwersja HU -> modul Younga.
        slice_hu = scale_to_hu(slice_gray)
        rho_qct = a * slice_hu - b
        rho_ash = (c + rho_qct) / d
        rho_app = rho_ash / 0.6

        E_map = e * (np.maximum(rho_app, 0) ** f)
        E_map_cortex = np.where(mask_cortex, E_map, 0)

        y, x = np.indices((h, w))
        y_c = np.mean(y[mask_cortex])
        x_c = np.mean(x[mask_cortex])
        y_rel = (y - y_c) * pixel_size
        x_rel = (x - x_c) * pixel_size
        r_sq = x_rel**2 + y_rel**2

        I_sag = np.sum(y_rel[mask_cortex] ** 2) * (pixel_size**2)
        IE_sag = np.sum(E_map_cortex[mask_cortex] * (y_rel[mask_cortex] ** 2)) * (pixel_size**2)

        I_front = np.sum(x_rel[mask_cortex] ** 2) * (pixel_size**2)
        IE_front = np.sum(E_map_cortex[mask_cortex] * (x_rel[mask_cortex] ** 2)) * (pixel_size**2)

        J_polar = np.sum(r_sq[mask_cortex]) * (pixel_size**2)
        JE_polar = np.sum(E_map_cortex[mask_cortex] * r_sq[mask_cortex]) * (pixel_size**2)

        y_max = np.max(np.abs(y_rel[mask_cortex])) if np.any(mask_cortex) else 1
        S_sag = I_sag / y_max
        SE_sag = IE_sag / y_max

        results.append(
            {
                "slice": i,
                "area_cortex_mm2": area_cortex,
                "cortical_ratio": cortical_ratio,
                "E_mean_GPa": np.mean(E_map_cortex[mask_cortex]),
                "I_sagittal_mm4": I_sag,
                "IE_sagittal_GPa_mm4": IE_sag,
                "I_frontal_mm4": I_front,
                "IE_frontal_GPa_mm4": IE_front,
                "J_polar_mm4": J_polar,
                "JE_polar_GPa_mm4": JE_polar,
                "Section_Modulus_S": S_sag,
                "Weighted_Section_Modulus_SE": SE_sag,
            }
        )

    return pd.DataFrame(results)


def load_masks(batch_name):
    im_sample = Image.open(batch_name[0])
    w, h = im_sample.size

    masks = np.zeros((len(batch_name), h, w), dtype=np.uint8)

    for i, imfile in enumerate(tqdm(batch_name, desc="Wczytywanie masek")):
        with Image.open(imfile) as inputslice:
            masks[i, :, :] = np.array(inputslice)

    return masks


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cala_kosc_files = sorted(glob.glob(os.path.join(script_dir, "Cala_kosc", "*.tif")))
    zbita_files = sorted(glob.glob(os.path.join(script_dir, "Tkanka_zbita", "*.tif")))

    if not cala_kosc_files or not zbita_files:
        print("Nie znaleziono plikow w folderach Cala_kosc albo Tkanka_zbita")
        exit()

    if len(cala_kosc_files) != len(zbita_files):
        print(f"nierowna liczba plikow: Cala kosc: {len(cala_kosc_files)}, Kora: {len(zbita_files)}")
        exit()

    rib = stack(cala_kosc_files)
    main_bone_shell = load_masks(zbita_files)
    full_bone_solid = load_masks(cala_kosc_files)

    df_stats = calculate_mechanics(rib, main_bone_shell, full_bone_solid, pixel_size=0.15)

    if not df_stats.empty:
        output_file = os.path.join(script_dir, "wyniki_biomechaniczne.csv")
        df_stats.to_csv(output_file, index=False)

        cols_to_show = ["cortical_ratio", "E_mean_GPa", "IE_sagittal_GPa_mm4"]
        existing_cols = [c for c in cols_to_show if c in df_stats.columns]

        if existing_cols:
            print("\nSrednie parametry dla probki:")
            print(df_stats[existing_cols].mean())
        else:
            print("Blad: Obliczone dane nie zawieraja wymaganych kolumn.")
    else:
        print("Funkcja calculate_mechanics zwrocila pusty zestaw danych")
