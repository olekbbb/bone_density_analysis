import os
import glob
import numpy as np
import cv2
import napari
import vedo
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from skimage import measure, morphology
from skimage.transform import resize
from scipy import ndimage
from vedo.applications import FreeHandCutPlotter

# --- 1. WCZYTYWANIE OBRAZÓW ---
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


# --- 2. SEGMENTACJA GMM ---
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

    last_successful_mask = None 

    for i in tqdm(range(z), desc='Segmentacja strukturalna'):
        roi = roi_stack[i]
        
        if np.max(roi) < (ref_bone_mean * 0.75):
            last_successful_mask = None 
            continue
            
        try:
            gm = GaussianMixture(n_components=3, means_init=ref_means, random_state=42).fit(roi.reshape((-1, 1)))
            means = gm.means_.flatten()
            bone_class = np.argmax(means)
            
            if means[bone_class] < (ref_bone_mean * 0.6):
                if last_successful_mask is not None:
                    bone_mask[i, r_start:r_end, c_start:c_end] = last_successful_mask
                continue
                
            probs = gm.predict_proba(roi.reshape((-1, 1)))[:, bone_class].reshape(h_roi, w_roi)
            seeds = (probs > 0.5).astype('uint8')
            potential = (probs > 0.3).astype('uint8')
            mask_connected = cv2.dilate(seeds, np.ones((3,3), np.uint8), iterations=2)
            mask_refined = np.where((mask_connected == 1) & (potential == 1), 255, 0).astype('uint8')
            
            bone_mask[i, r_start:r_end, c_start:c_end] = mask_refined
            last_successful_mask = mask_refined 
            
        except Exception:
            if last_successful_mask is not None:
                bone_mask[i, r_start:r_end, c_start:c_end] = last_successful_mask
            continue
            
    print("\n⏳ Usuwanie szumów: izolacja największego elementu...")
    binary_3d = (bone_mask > 0).astype(np.uint8)
    labels = measure.label(binary_3d, connectivity=3)
    props = measure.regionprops(labels)
    
    if props:
        largest_label = max(props, key=lambda p: p.area).label
        bone_mask = (labels == largest_label).astype(np.uint8) * 255

    return volume, bone_mask


# --- 3. INTERAKTYWNE CIĘCIE (CHMURA PUNKTÓW 1:1) ---
def extract_main_bone_interactive(bone_mask_3d):
    print("\n" + "="*60)
    print(" 🚀 SZYBKIE ODCINANIE W 3D (PEŁNY ROZMIAR 1:1):")
    print(" 1. Obróć bryłę lewym przyciskiem myszy.")
    print(" 2. Wciśnij klawisz 'r' lub 'x', aby zacząć rysować lasso.")
    print(" 3. Obrysuj niechciany element (tniemy bezpośrednio na oryginale!).")
    print(" 4. Gdy skończysz, wciśnij klawisz 'q' lub zamknij okno X.")
    print("="*60 + "\n")

    print("⏳ Przygotowanie maski (Oryginalna skala)...")
    binary_mask = (bone_mask_3d > 0).astype(np.uint8)
    labels_pre = measure.label(binary_mask, connectivity=3)
    props_pre = measure.regionprops(labels_pre)
    
    if not props_pre:
        return bone_mask_3d
        
    largest_pre = max(props_pre, key=lambda p: p.area).label
    clean_initial_mask = (labels_pre == largest_pre).astype(np.uint8)

    z_idx, y_idx, x_idx = np.where(clean_initial_mask > 0)
    coords = np.column_stack((x_idx, y_idx, z_idx))
    
    pts = vedo.Points(coords, r=2).c("gold").lighting("plastic")

    plotter = FreeHandCutPlotter(pts)
    plotter.start()

    print("🔄 Okno zamknięte. Odtwarzanie oryginalnej maski...")
    
    if hasattr(plotter, "mesh") and plotter.mesh is not None:
        surviving_coords = plotter.mesh.vertices
    else:
        surviving_coords = pts.vertices
    
    if len(surviving_coords) == 0:
        print("⚠️ Wymazano wszystko lub okno zamknięto bez zmian. Zwracam oryginał.")
        return clean_initial_mask * 255

    surviving_coords = np.round(surviving_coords).astype(int)
    
    valid = (
        (surviving_coords[:, 0] >= 0) & (surviving_coords[:, 0] < clean_initial_mask.shape[2]) &
        (surviving_coords[:, 1] >= 0) & (surviving_coords[:, 1] < clean_initial_mask.shape[1]) &
        (surviving_coords[:, 2] >= 0) & (surviving_coords[:, 2] < clean_initial_mask.shape[0])
    )
    surviving_coords = surviving_coords[valid]

    grid_data = np.zeros_like(clean_initial_mask, dtype=bool)
    grid_data[surviving_coords[:, 2], surviving_coords[:, 1], surviving_coords[:, 0]] = True

    final_labels = measure.label(grid_data, connectivity=3)
    props_final = measure.regionprops(final_labels)
    
    if not props_final:
        return clean_initial_mask * 255
        
    largest_final = max(props_final, key=lambda p: p.area)
    print("✅ Sukces! Odcięte elementy zostały usunięte z matrycy (piksel do piksela).")
    
    return (final_labels == largest_final.label).astype(np.uint8) * 255


# --- 4. ZALEWANIE WNĘTRZA KOŚCI (FILL VOLUME) ---
def fill_bone_volume(bone_mask_3d, closing_radius=8):
    z, h, w = bone_mask_3d.shape
    mask_bool = bone_mask_3d > 0
    
    if not np.any(mask_bool):
        return np.zeros_like(bone_mask_3d)
        
    print("\n💧 Rozpoczynam zalewanie objętości (wypełnianie szpiku)...")
    print("Krok 1/4: Skalowanie wolumenu w dół (optymalizacja)")
    small_vol = resize(mask_bool.astype(float),
                       (z // 4, h // 2, w // 2),
                       order=0, preserve_range=True, anti_aliasing=False) > 0.5

    print("Krok 2/4: Zalewanie bryły w 3D (na mniejszej skali)")
    struct = morphology.ball(closing_radius // 2)
    sealed = morphology.binary_closing(small_vol, struct)
    filled = ndimage.binary_fill_holes(sealed)
    
    print("Krok 3/4: Przywracanie oryginalnej rozdzielczości")
    full_filled = resize(filled.astype(float), (z, h, w),
                  order=0, preserve_range=True, anti_aliasing=False) > 0.5

    print("Krok 4/4: Wygładzanie krawędzi (Erozja)")
    final_mask = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    struct_2d = morphology.disk(2)

    for i in range(z):
        if np.any(full_filled[i]):
            res = morphology.binary_erosion(full_filled[i], struct_2d)
            final_mask[i] = res.astype(np.uint8) * 255
            
    print("✅ Zalewanie zakończone!")
    return final_mask


# --- 5. EKSPORT WYMASKOWANYCH OBRAZÓW ---
def export_masked_images(original_volume, mask_volume, output_folder, prefix="slice"):
    """
    Mnoży oryginalne zdjęcia przez maskę (usuwa tło) i zapisuje je jako pliki .tif
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"\n📁 Zapisywanie wymaskowanych zdjęć do folderu: {output_folder} ...")
    z_slices = original_volume.shape[0]
    
    for i in tqdm(range(z_slices), desc=f'Eksport: {prefix}'):
        orig_slice = original_volume[i]
        mask_slice = mask_volume[i]
        
        # Gdzie maska > 0 (jest kość), tam zostawiamy oryginalny piksel. W przeciwnym razie czarne tło (0).
        masked_slice = np.where(mask_slice > 0, orig_slice, 0)
        
        # Konwersja na format 8-bitowy (standard dla obrazów)
        masked_uint8 = np.clip(masked_slice, 0, 255).astype(np.uint8)
        
        # Zapis za pomocą biblioteki PIL
        img = Image.fromarray(masked_uint8)
        filename = os.path.join(output_folder, f"{prefix}_{i:04d}.tif")
        img.save(filename)
    print("✅ Eksport zakończony sukcesem!")


# --- 6. GŁÓWNA LOGIKA WYKONAWCZA ---
if __name__ == "__main__":
    infiles = sorted(glob.glob(r'C:\Users\Supri\Desktop\bone_34_476\*.tif'))
    
    CACHE_SEGMENTATION = "1_initial_segmented_mask.npy"
    CACHE_CLEANED = "2_cleaned_shell_mask.npy"
    
    # Foldery docelowe dla gotowych obrazów
    FOLDER_KORA = r"C:\Users\Supri\Desktop\Tkanka_zbita"
    FOLDER_BRYLA = r"C:\Users\Supri\Desktop\Cala_kosc"
    
    if not infiles:
        print("Nie znaleziono obrazów. Sprawdź ścieżkę do plików!")
    else:
        rib = stack(infiles)
        COORDS = (100, 600, 100, 600)
        
        # ETAP 1: Segmentacja
        if os.path.exists(CACHE_SEGMENTATION):
            print(f"\n💾 Znaleziono zapisaną segmentację: {CACHE_SEGMENTATION}. Wczytywanie...")
            initial_mask = np.load(CACHE_SEGMENTATION)
        else:
            print("\n⏳ Brak bufora segmentacji. Uruchamiam GMM...")
            _, initial_mask = segmentate(rib, roi_coords=COORDS)
            np.save(CACHE_SEGMENTATION, initial_mask)
            print("✅ Segmentacja zapisana na dysku.")

        # ETAP 2: Czyszczenie (Vedo)
        if os.path.exists(CACHE_CLEANED):
            print(f"\n💾 Znaleziono maskę po cięciu w Vedo ({CACHE_CLEANED}). Wczytywanie...")
            main_bone_shell = np.load(CACHE_CLEANED)
            print("✅ Pominięto etap Vedo!")
        else:
            print("\n⏳ Brak wyczyszczonej maski. Uruchamiam Vedo...")
            main_bone_shell = extract_main_bone_interactive(initial_mask)
            print(f"\n💾 Zapisywanie oczyszczonej tkanki zbitej do: {CACHE_CLEANED}...")
            np.save(CACHE_CLEANED, main_bone_shell)
            print("✅ Maska po cięciu zapisana.")

        # ETAP 3: Wypełnianie (Solid) - BEZ ZAPISU DO NPY
        print("\n⏳ Uruchamiam wypełnianie objętości...")
        full_bone_solid = fill_bone_volume(main_bone_shell, closing_radius=30)

        # ETAP 4: Eksport gotowych obrazów do folderów na komputerze
        export_masked_images(rib, main_bone_shell, FOLDER_KORA, prefix="kora")
        export_masked_images(rib, full_bone_solid, FOLDER_BRYLA, prefix="bryla")

        # ETAP 5: Wyświetlanie w Napari
        print("\n🎨 Otwieram widok Napari...")
        viewer = napari.Viewer()
        viewer.add_image(rib, name='Oryginał (Pełny)', blending='additive')
        
        viewer.add_labels(initial_mask, name='1. Wstępna segmentacja', visible=False)
        viewer.add_labels(full_bone_solid, name='3. Pełna Bryła (Wypełniona)', opacity=0.3, visible=True)
        viewer.add_labels(main_bone_shell, name='2. Tkanka Zbita (Kora)', opacity=1.0, visible=True)
        
        napari.run()