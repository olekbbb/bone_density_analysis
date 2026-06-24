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


def stack(batch_name):
    im_sample = Image.open(batch_name[0])
    orig_w, orig_h = im_sample.size
    new_w, new_h = orig_w // 4, orig_h // 4
    imgs = np.zeros((len(batch_name), new_h, new_w), dtype=np.float32)

    for i, imfile in enumerate(tqdm(batch_name, desc='Wczytywanie obrazów')):
        with Image.open(imfile) as inputslice:
            resized_slice = inputslice.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            imgs[i, :, :] = np.array(resized_slice)
    return imgs


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

    for i in tqdm(range(z), desc='Segmentacja'):
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

    roi_mask = bone_mask[:, r_start:r_end, c_start:c_end]
    binary_3d_roi = (roi_mask > 0).astype(np.uint8)
    
    labels_roi = measure.label(binary_3d_roi, connectivity=3)
    props = measure.regionprops(labels_roi)
    
    if props:
        largest_label = max(props, key=lambda p: p.area).label
        clean_roi_mask = (labels_roi == largest_label).astype(np.uint8) * 255
        bone_mask.fill(0)
        bone_mask[:, r_start:r_end, c_start:c_end] = clean_roi_mask

    return volume, bone_mask


def extract_main_bone_interactive(bone_mask_3d):
    coords_z, coords_y, coords_x = np.where(bone_mask_3d > 0)
    if len(coords_z) == 0:
        return bone_mask_3d

    z0, z1 = coords_z.min(), coords_z.max() + 1
    y0, y1 = coords_y.min(), coords_y.max() + 1
    x0, x1 = coords_x.min(), coords_x.max() + 1

    cropped_mask = bone_mask_3d[z0:z1, y0:y1, x0:x1]
    labels_pre = measure.label(cropped_mask > 0, connectivity=3)
    props_pre = measure.regionprops(labels_pre)
    
    clean_initial_mask = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    if props_pre:
        largest_pre = max(props_pre, key=lambda p: p.area).label
        cropped_clean = (labels_pre == largest_pre).astype(np.uint8)
        clean_initial_mask[z0:z1, y0:y1, x0:x1] = cropped_clean
    else:
        return bone_mask_3d

    z_idx, y_idx, x_idx = np.where(clean_initial_mask > 0)
    coords = np.column_stack((x_idx, y_idx, z_idx))
    
    pts = vedo.Points(coords, r=2).c("gold").lighting("plastic")
    plotter = FreeHandCutPlotter(pts)
    plotter.start()
    
    if hasattr(plotter, "mesh") and plotter.mesh is not None:
        surviving_coords = plotter.mesh.vertices
    else:
        surviving_coords = pts.vertices
    
    if len(surviving_coords) == 0:
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

    c_z, c_y, c_x = np.where(grid_data)
    if len(c_z) == 0:
        return clean_initial_mask * 255
        
    cz0, cz1 = c_z.min(), c_z.max() + 1
    cy0, cy1 = c_y.min(), c_y.max() + 1
    cx0, cx1 = c_x.min(), c_x.max() + 1
    
    cropped_grid = grid_data[cz0:cz1, cy0:cy1, cx0:cx1]
    final_labels = measure.label(cropped_grid, connectivity=3)
    props_final = measure.regionprops(final_labels)
    
    if not props_final:
        return clean_initial_mask * 255
        
    largest_final = max(props_final, key=lambda p: p.area).label
    cropped_final_clean = (final_labels == largest_final).astype(np.uint8) * 255
    
    final_output = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    final_output[cz0:cz1, cy0:cy1, cx0:cx1] = cropped_final_clean
    
    return final_output


def fill_bone_volume(bone_mask_3d, closing_radius=8):
    z, h, w = bone_mask_3d.shape
    mask_bool = bone_mask_3d > 0
    
    if not np.any(mask_bool):
        return np.zeros_like(bone_mask_3d)
        
    print("Krok 1/4: Skalowanie wolumenu w dół")
    small_vol = resize(mask_bool.astype(float),
                       (z // 4, h // 2, w // 2),
                       order=0, preserve_range=True, anti_aliasing=False) > 0.5

    print("Krok 2/4: Zalewanie bryły w 3D")
    struct = morphology.ball(closing_radius // 2)
    sealed = morphology.binary_closing(small_vol, struct)
    filled = ndimage.binary_fill_holes(sealed)
    
    print("Krok 3/4: Przywracanie oryginalnej rozdzielczości")
    full_filled = resize(filled.astype(float), (z, h, w),
                  order=0, preserve_range=True, anti_aliasing=False) > 0.5

    print("Krok 4/4: Wygładzanie krawędzi")
    final_mask = np.zeros_like(bone_mask_3d, dtype=np.uint8)
    struct_2d = morphology.disk(2)

    for i in range(z):
        if np.any(full_filled[i]):
            res = morphology.binary_erosion(full_filled[i], struct_2d)
            final_mask[i] = res.astype(np.uint8) * 255
            
    return final_mask


def export_masked_images(original_volume, mask_volume, output_folder, prefix="slice"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Zapisywanie wymaskowanych zdjęć do folderu: {output_folder} ...")
    z_slices = original_volume.shape[0]
    
    for i in tqdm(range(z_slices), desc=f'Eksport: {prefix}'):
        orig_slice = original_volume[i]
        mask_slice = mask_volume[i]
        
        masked_slice = np.where(mask_slice > 0, orig_slice, 0)
        masked_uint8 = np.clip(masked_slice, 0, 255).astype(np.uint8)
        img = Image.fromarray(masked_uint8)
        filename = os.path.join(output_folder, f"{prefix}_{i:04d}.tif")
        img.save(filename)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    infiles = sorted(glob.glob(os.path.join(script_dir, "bone_34_476", "*.tif")))
    
    CACHE_SEGMENTATION = "1_initial_segmented_mask.npy"
    CACHE_CLEANED = "2_cleaned_shell_mask.npy"
    
    FOLDER_KORA = os.path.join(script_dir, "Tkanka_zbita")
    FOLDER_BRYLA = os.path.join(script_dir, "Cala_kosc")
    
    if not infiles:
        print("Nie znaleziono obrazów")
    else:
        rib = stack(infiles)
        COORDS = (50, 300, 50, 300)
        
        if os.path.exists(CACHE_SEGMENTATION):
            print(f"\n Znaleziono zapisaną segmentację: {CACHE_SEGMENTATION}")
            initial_mask = np.load(CACHE_SEGMENTATION)
        else:
            _, initial_mask = segmentate(rib, roi_coords=COORDS)
            np.save(CACHE_SEGMENTATION, initial_mask) 

        if os.path.exists(CACHE_CLEANED):
            print(f"Znaleziono maskę po cięciu w Vedo ({CACHE_CLEANED})")
            main_bone_shell = np.load(CACHE_CLEANED)
        else:
            main_bone_shell = extract_main_bone_interactive(initial_mask)
            np.save(CACHE_CLEANED, main_bone_shell)
            
        full_bone_solid = fill_bone_volume(main_bone_shell, closing_radius=30)

        export_masked_images(rib, main_bone_shell, FOLDER_KORA, prefix="kora")
        export_masked_images(rib, full_bone_solid, FOLDER_BRYLA, prefix="bryla")

        viewer = napari.Viewer()
        viewer.add_image(rib, name='Oryginał (Pełny)', blending='additive')
        
        viewer.add_labels(initial_mask, name='1. Wstępna segmentacja', visible=False)
        viewer.add_labels(full_bone_solid, name='3. Pełna Bryła (Wypełniona)', opacity=0.3, visible=True)
        viewer.add_labels(main_bone_shell, name='2. Tkanka Zbita (Kora)', opacity=1.0, visible=True)
        
        napari.run()
