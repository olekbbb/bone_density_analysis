import glob
import numpy as np
import cv2
import napari
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from skimage import measure, morphology
from scipy import ndimage
from skimage.transform import resize

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

def extract_main_bone(bone_mask_3d, roi_coords=(100, 600, 100, 600)):

    r_s, r_e, c_s, c_e = roi_coords
    clean_mask = np.zeros_like(bone_mask_3d)
    
    roi_area = (bone_mask_3d[:, r_s:r_e, c_s:c_e] > 0).astype(bool)
    if not np.any(roi_area):
        return clean_mask

    bridge_struct = morphology.ball(5) 
    roi_connected = morphology.binary_closing(roi_area, bridge_struct)

    labels = measure.label(roi_connected, connectivity=3)
    props = measure.regionprops(labels)
    
    if props:
        main_label = max(props, key=lambda x: x.area).label
        connected_bone = (labels == main_label)
        
        final_bone_roi_bool = np.logical_or(roi_area, morphology.binary_erosion(connected_bone, morphology.ball(2)))
        final_bone_roi_bool = np.logical_and(final_bone_roi_bool, connected_bone)
        
        final_bone_roi = (final_bone_roi_bool.astype(np.uint8)) * 255
        for i in range(final_bone_roi.shape[0]):
            final_bone_roi[i] = cv2.medianBlur(final_bone_roi[i], 5)

        clean_mask[:, r_s:r_e, c_s:c_e] = final_bone_roi
    
    return clean_mask

def fill_bone_volume(bone_mask_3d, closing_radius=8):
    z, h, w = bone_mask_3d.shape
    mask_bool = bone_mask_3d > 0
    
    if not np.any(mask_bool):
        return np.zeros_like(bone_mask_3d)
    print("Krok 1/4: Skalowanie wolumenu w dół")
    small_vol = resize(mask_bool.astype(float), 
                       (z // 4, h // 2, w // 2), 
                       order=0, preserve_range=True, anti_aliasing=False) > 0.5

    print("Krok 2/4: Zalewanie bryły w 3D (na mniejszej skali)")
    struct = morphology.ball(closing_radius // 2) # Skalujemy też promień kuli
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
if __name__ == "__main__":
    
    infiles = sorted(glob.glob(r'C:\Users\Supri\Desktop\bone_34_476\*.tif'))
    rib = stack(infiles)
    COORDS = (100, 600, 100, 600)
    full_vol, initial_mask = segmentate(rib, roi_coords=COORDS)
    main_bone_shell = extract_main_bone(initial_mask, roi_coords=COORDS)
    full_bone_solid = fill_bone_volume(main_bone_shell, closing_radius=40)
    view(full_vol, main_bone_shell, full_bone_solid)