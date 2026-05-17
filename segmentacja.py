import argparse
import glob
import json
from pathlib import Path
import numpy as np
import cv2
import napari
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from skimage import measure, morphology
from scipy import ndimage
from skimage.transform import resize
import pandas as pd

DEFAULT_SEGMENTATION_FILE = "gotowa_segmentacja.npz"
DEFAULT_RESULTS_CSV = "wyniki_biomechaniczne.csv"
DEFAULT_DASHBOARD_DATA = "frontend/wyniki_data.json"
DEFAULT_BLOCK_DEPTH = 48
DEFAULT_BLOCK_OVERLAP = 12

def stack(batch_name):
    im_sample = Image.open(batch_name[0])
    orig_w, orig_h = im_sample.size
    new_w, new_h = orig_w // 2, orig_h // 2
    
    imgs = np.zeros((len(batch_name), new_h, new_w), dtype=np.float32)
    
    for i, imfile in enumerate(tqdm(batch_name, desc='Wczytywanie obrazów')):
        with Image.open(imfile) as inputslice:
            resized_slice = inputslice.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            imgs[i, :, :] = np.array(resized_slice)
            
    return imgs

def view(stack, mask_kora, mask_bryla):
    viewer = napari.Viewer()
    viewer.add_image(stack, name='Oryginał (Pełny)')
    viewer.add_labels(mask_kora, name='Tkanka Zbita (Kora)')
    viewer.add_labels(mask_bryla, name='Pełna Bryła (Zalana)', opacity=0.3)
    napari.run()

def save_segmentation(output_path, volume, mask_kora, mask_bryla, roi_coords=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "volume": volume,
        "mask_kora": mask_kora,
        "mask_bryla": mask_bryla,
    }
    if roi_coords is not None:
        payload["roi_coords"] = np.array(roi_coords, dtype=np.int32)

    np.savez_compressed(output_path, **payload)
    print(f"Gotowa segmentacja zapisana do {output_path}")
    return output_path

def load_segmentation(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Nie znaleziono zapisanej segmentacji: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        required_arrays = ("volume", "mask_kora", "mask_bryla")
        missing_arrays = [name for name in required_arrays if name not in data]
        if missing_arrays:
            missing = ", ".join(missing_arrays)
            raise ValueError(f"Plik segmentacji nie zawiera wymaganych tablic: {missing}")

        volume = data["volume"]
        mask_kora = data["mask_kora"]
        mask_bryla = data["mask_bryla"]
        metadata = {}
        if "roi_coords" in data:
            metadata["roi_coords"] = tuple(int(v) for v in data["roi_coords"])

    print(f"Wczytano gotową segmentację z {input_path}")
    return volume, mask_kora, mask_bryla, metadata

def segmentate(volume, roi_coords=(100, 600, 100, 600)):
    z, h_orig, w_orig = volume.shape
    r_start, r_end, c_start, c_end = roi_coords
    roi_stack = volume[:, r_start:r_end, c_start:c_end]
    bone_mask = np.zeros_like(volume, dtype=np.uint8)
    
    h_roi, w_roi = r_end - r_start, c_end - c_start
    
    brightest_idx = np.argmax(np.max(roi_stack, axis=(1, 2)))
    ref_roi = roi_stack[brightest_idx]
    gm_ref = GaussianMixture(n_components=3, random_state=42).fit(ref_roi.reshape((-1, 1)))
    ref_means = np.sort(gm_ref.means_.flatten()).reshape(-1, 1)
    ref_bone_mean = ref_means[2][0]

    for i in tqdm(range(z), desc='Segmentacja strukturalna'):
        roi = roi_stack[i]
        if np.max(roi) < (ref_bone_mean * 0.5): 
            continue
            
        try:
            gm = GaussianMixture(n_components=3, means_init=ref_means, random_state=42).fit(roi.reshape((-1, 1)))
            means = gm.means_.flatten()
            bone_class = np.argmax(means)
            
            if means[bone_class] < (ref_bone_mean):
                continue

            probs = gm.predict_proba(roi.reshape((-1, 1)))[:, bone_class].reshape(h_roi, w_roi)
            
            seeds = (probs > 0.7).astype('uint8')
            potential = (probs > 0.4).astype('uint8')
            mask_connected = cv2.dilate(seeds, np.ones((3,3), np.uint8), iterations=2)
            mask_refined = np.where((mask_connected == 1) & (potential == 1), 255, 0).astype('uint8')

            bone_mask[i, r_start:r_end, c_start:c_end] = mask_refined
            
        except Exception:
            continue

    return volume, bone_mask

def iter_z_blocks(depth, block_depth, overlap):
    block_depth = max(1, int(block_depth))
    overlap = max(0, int(overlap))
    start = 0
    while start < depth:
        end = min(start + block_depth, depth)
        read_start = max(0, start - overlap)
        read_end = min(depth, end + overlap)
        yield start, end, read_start, read_end
        start = end

def count_z_blocks(depth, block_depth):
    block_depth = max(1, int(block_depth))
    return (depth + block_depth - 1) // block_depth

def extract_main_bone(
    bone_mask_3d,
    roi_coords=(100, 600, 100, 600),
    block_depth=DEFAULT_BLOCK_DEPTH,
    block_overlap=DEFAULT_BLOCK_OVERLAP,
):

    r_s, r_e, c_s, c_e = roi_coords
    clean_mask = np.zeros_like(bone_mask_3d)

    if not np.any(bone_mask_3d[:, r_s:r_e, c_s:c_e]):
        return clean_mask

    bridge_struct = morphology.ball(5)
    erosion_struct = morphology.ball(2)
    z_depth = bone_mask_3d.shape[0]

    for start, end, read_start, read_end in tqdm(
        iter_z_blocks(z_depth, block_depth, block_overlap),
        total=count_z_blocks(z_depth, block_depth),
        desc="Czyszczenie głównej kości blokami",
    ):
        roi_area = (bone_mask_3d[read_start:read_end, r_s:r_e, c_s:c_e] > 0)
        if not np.any(roi_area):
            continue

        roi_connected = morphology.binary_closing(roi_area, bridge_struct)
        labels = measure.label(roi_connected, connectivity=3)
        counts = np.bincount(labels.ravel())
        if len(counts) <= 1:
            continue

        counts[0] = 0
        main_label = int(np.argmax(counts))
        connected_bone = (labels == main_label)

        eroded = morphology.binary_erosion(connected_bone, erosion_struct)
        final_bone_roi_bool = np.logical_or(roi_area, eroded)
        final_bone_roi_bool = np.logical_and(final_bone_roi_bool, connected_bone)

        core_start = start - read_start
        core_end = core_start + (end - start)
        final_bone_roi = (final_bone_roi_bool[core_start:core_end].astype(np.uint8)) * 255
        for i in range(final_bone_roi.shape[0]):
            final_bone_roi[i] = cv2.medianBlur(final_bone_roi[i], 5)

        clean_mask[start:end, r_s:r_e, c_s:c_e] = final_bone_roi

    return clean_mask

def fill_holes_2d(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)

def fill_bone_volume_2d(bone_mask_3d, closing_radius=40):
    if not np.any(bone_mask_3d):
        return np.zeros_like(bone_mask_3d)

    print("Zalewanie bryły przekrój po przekroju (tryb oszczędzania RAM)")
    final_mask = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(1, closing_radius * 2 + 1), max(1, closing_radius * 2 + 1)),
    )
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for i in tqdm(range(bone_mask_3d.shape[0]), desc="Zalewanie bryły 2D"):
        current = (bone_mask_3d[i] > 0).astype(np.uint8) * 255
        if not np.any(current):
            continue

        closed = cv2.morphologyEx(current, cv2.MORPH_CLOSE, close_kernel)
        filled = fill_holes_2d(closed)
        final_mask[i] = cv2.erode(filled, erode_kernel, iterations=1)

    return final_mask

def fill_bone_volume_3d_blocks(
    bone_mask_3d,
    closing_radius=8,
    block_depth=DEFAULT_BLOCK_DEPTH,
    block_overlap=None,
):
    z, h, w = bone_mask_3d.shape

    if not np.any(bone_mask_3d):
        return np.zeros_like(bone_mask_3d)

    if block_overlap is None:
        block_overlap = max(DEFAULT_BLOCK_OVERLAP, closing_radius)

    print("Zalewanie bryły w blokach Z")
    final_mask = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    struct_3d = morphology.ball(max(1, closing_radius // 2))
    struct_2d = morphology.disk(2)

    for start, end, read_start, read_end in tqdm(
        iter_z_blocks(z, block_depth, block_overlap),
        total=count_z_blocks(z, block_depth),
        desc="Zalewanie bryły blokami",
    ):
        block = bone_mask_3d[read_start:read_end] > 0
        if not np.any(block):
            continue

        small_shape = (
            max(1, block.shape[0] // 4),
            max(1, h // 2),
            max(1, w // 2),
        )
        small_vol = resize(
            block.astype(np.float32),
            small_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5

        sealed = morphology.binary_closing(small_vol, struct_3d)
        filled = ndimage.binary_fill_holes(sealed)
        full_filled = resize(
            filled.astype(np.float32),
            block.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5

        core_start = start - read_start
        core_end = core_start + (end - start)
        full_filled_core = full_filled[core_start:core_end]

        for i in range(full_filled_core.shape[0]):
            if np.any(full_filled_core[i]):
                res = morphology.binary_erosion(full_filled_core[i], struct_2d)
                final_mask[start + i] = res.astype(np.uint8) * 255
            
    return final_mask

def fill_bone_volume(
    bone_mask_3d,
    closing_radius=40,
    block_depth=DEFAULT_BLOCK_DEPTH,
    block_overlap=None,
    mode="2d",
):
    if mode == "3d-block":
        return fill_bone_volume_3d_blocks(
            bone_mask_3d,
            closing_radius=closing_radius,
            block_depth=block_depth,
            block_overlap=block_overlap,
        )

    return fill_bone_volume_2d(bone_mask_3d, closing_radius=closing_radius)

def calculate_mechanics(volume, mask_kora, mask_bryla, pixel_size=0.5):
    z, h, w = volume.shape
    results = []

    #params from article 
    a, b = 0.000785524, 0.004277819
    c, d = 0.079, 0.877
    e, f = 3.891, 2.39

    pixel_min, pixel_max = np.min(volume), np.max(volume)
    hu_min, hu_max = -600, 1500 # Przykładowy zakres HU

    def scale_to_hu(pixel_val):
        #interpolacja liniowa
        return hu_min + (pixel_val - pixel_min) * (hu_max - hu_min) / (pixel_max - pixel_min)

    for i in tqdm(range(z), desc='Analiza przekrojów'):
        slice_gray = volume[i]
        mask_cortex = mask_kora[i] > 0
        mask_total = mask_bryla[i] > 0

        if not np.any(mask_cortex):
            continue

        area_cortex = np.sum(mask_cortex) * (pixel_size**2)
        area_total = np.sum(mask_total) * (pixel_size**2)
        cortical_ratio = area_cortex / area_total if area_total > 0 else 0

        #konwersja HU -> moduł younga
        slice_hu = scale_to_hu(slice_gray)
        rho_qct = a * slice_hu - b        
        rho_ash = (c + rho_qct) / d
        rho_app = rho_ash / 0.6
        #maska dla tkanki kostnej
        E_map = e * (np.maximum(rho_app, 0)**f)
        E_map_cortex = np.where(mask_cortex, E_map, 0)
        y, x = np.indices((h, w))
        y_c = np.mean(y[mask_cortex])
        x_c = np.mean(x[mask_cortex])
        y_rel = (y - y_c) * pixel_size
        x_rel = (x - x_c) * pixel_size
        r_sq = x_rel**2 + y_rel**2

        #I_sagittal (wokół osi ML - z w artykule) = ∫ y^2 dA
        I_sag = np.sum(y_rel[mask_cortex]**2) * (pixel_size**2)
        IE_sag = np.sum(E_map_cortex[mask_cortex] * (y_rel[mask_cortex]**2)) * (pixel_size**2)

        #I_frontal (wokół osi AP - y w artykule) = ∫ x^2 dA
        I_front = np.sum(x_rel[mask_cortex]**2) * (pixel_size**2)
        IE_front = np.sum(E_map_cortex[mask_cortex] * (x_rel[mask_cortex]**2)) * (pixel_size**2)

        #polar moment
        J_polar = np.sum(r_sq[mask_cortex]) * (pixel_size**2)
        JE_polar = np.sum(E_map_cortex[mask_cortex] * r_sq[mask_cortex]) * (pixel_size**2)

        #Moduł przekroju (Section Modulus) S = I / y_max
        y_max = np.max(np.abs(y_rel[mask_cortex])) if np.any(mask_cortex) else 1
        x_max = np.max(np.abs(x_rel[mask_cortex])) if np.any(mask_cortex) else 1
        
        S_sag = I_sag / y_max
        SE_sag = IE_sag / y_max

        results.append({
            'slice': i,
            'area_cortex_mm2': area_cortex,
            'cortical_ratio': cortical_ratio,
            'E_mean_GPa': np.mean(E_map_cortex[mask_cortex]),
            'I_sagittal_mm4': I_sag,
            'IE_sagittal_GPa_mm4': IE_sag,
            'I_frontal_mm4': I_front,
            'IE_frontal_GPa_mm4': IE_front,
            'J_polar_mm4': J_polar,
            'JE_polar_GPa_mm4': JE_polar,
            'Section_Modulus_S': S_sag,
            'Weighted_Section_Modulus_SE': SE_sag
        })

    return pd.DataFrame(results)

def export_dashboard_data(df_stats, output_path=DEFAULT_DASHBOARD_DATA):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_columns = [
        col for col in df_stats.columns
        if pd.api.types.is_numeric_dtype(df_stats[col])
    ]
    metric_columns = [col for col in numeric_columns if col != "slice"]
    summary = {
        col: {
            "mean": float(df_stats[col].mean()),
            "min": float(df_stats[col].min()),
            "max": float(df_stats[col].max()),
        }
        for col in metric_columns
    }
    payload = {
        "rows": df_stats.to_dict(orient="records"),
        "columns": list(df_stats.columns),
        "numericColumns": numeric_columns,
        "metricColumns": metric_columns,
        "summary": summary,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dane do frontendu zapisane do {output_path}")
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentacja kości, zapis gotowej segmentacji i szybkie uruchamianie viewer."
    )
    parser.add_argument(
        "--input-glob",
        default="data/bone_34_476/*.tif",
        help="Ścieżka glob do plików TIFF wejściowego stosu.",
    )
    parser.add_argument(
        "--segmentation-file",
        default=DEFAULT_SEGMENTATION_FILE,
        help="Plik .npz z gotową segmentacją.",
    )
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Wczytaj zapisaną segmentację i uruchom tylko viewer.",
    )
    parser.add_argument(
        "--force-segmentation",
        action="store_true",
        help="Przelicz segmentację nawet jeśli plik segmentacji już istnieje.",
    )
    parser.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Nie uruchamiaj viewer po zakończeniu obliczeń.",
    )
    parser.add_argument(
        "--dashboard-data",
        default=DEFAULT_DASHBOARD_DATA,
        help="Plik JSON używany przez frontend wyników.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("R_START", "R_END", "C_START", "C_END"),
        default=(100, 600, 100, 600),
        help="Współrzędne ROI: r_start r_end c_start c_end.",
    )
    parser.add_argument(
        "--block-depth",
        type=int,
        default=DEFAULT_BLOCK_DEPTH,
        help="Liczba przekrojów Z przetwarzanych naraz w etapach po segmentacji strukturalnej.",
    )
    parser.add_argument(
        "--block-overlap",
        type=int,
        default=DEFAULT_BLOCK_OVERLAP,
        help="Nakładka przekrojów Z między blokami dla operacji morfologicznych.",
    )
    parser.add_argument(
        "--fill-overlap",
        type=int,
        default=40,
        help="Nakładka przekrojów Z dla zalewania bryły. Zmniejsz, jeśli nadal brakuje RAM.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=("2d", "3d-block"),
        default="2d",
        help="Sposób zalewania bryły: 2d jest najmniej pamięciożerny, 3d-block zachowuje blokowe operacje 3D.",
    )
    parser.add_argument(
        "--fill-radius",
        type=int,
        default=40,
        help="Promień domykania przy zalewaniu bryły.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    segmentation_path = Path(args.segmentation_file)
    COORDS = tuple(args.roi)

    if args.viewer_only:
        full_vol, main_bone_shell, full_bone_solid, _ = load_segmentation(segmentation_path)
        if not args.skip_viewer:
            view(full_vol, main_bone_shell, full_bone_solid)
        raise SystemExit(0)

    if segmentation_path.exists() and not args.force_segmentation:
        full_vol, main_bone_shell, full_bone_solid, _ = load_segmentation(segmentation_path)
        rib = full_vol
    else:
        infiles = sorted(glob.glob(args.input_glob))
        if not infiles:
            raise FileNotFoundError(f"Nie znaleziono plików wejściowych dla wzorca: {args.input_glob}")

        rib = stack(infiles)
        full_vol, initial_mask = segmentate(rib, roi_coords=COORDS)
        main_bone_shell = extract_main_bone(
            initial_mask,
            roi_coords=COORDS,
            block_depth=args.block_depth,
            block_overlap=args.block_overlap,
        )
        del initial_mask
        full_bone_solid = fill_bone_volume(
            main_bone_shell,
            closing_radius=args.fill_radius,
            block_depth=args.block_depth,
            block_overlap=args.fill_overlap,
            mode=args.fill_mode,
        )
        save_segmentation(segmentation_path, full_vol, main_bone_shell, full_bone_solid, roi_coords=COORDS)

    df_stats = calculate_mechanics(rib, main_bone_shell, full_bone_solid, pixel_size=0.15)
    if not df_stats.empty:
        df_stats.to_csv(DEFAULT_RESULTS_CSV, index=False)
        print(f"Wyniki zapisane do {DEFAULT_RESULTS_CSV}")
        export_dashboard_data(df_stats, args.dashboard_data)

        cols_to_show = ['cortical_ratio', 'E_mean_GPa', 'IE_sagittal_GPa_mm4']
        existing_cols = [c for c in cols_to_show if c in df_stats.columns]

        if existing_cols:
            print("\nŚrednie parametry dla próbki:")
            print(df_stats[existing_cols].mean())
        else:
            print("Błąd: Obliczone dane nie zawierają wymaganych kolumn.")
    else:
        print("Błąd: Funkcja calculate_mechanics zwróciła pusty zestaw danych.")

    if not args.skip_viewer:
        view(full_vol, main_bone_shell, full_bone_solid)
