# utils.py
import os
import numpy as np
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CMAP_DIR = os.path.join(REPO_ROOT, 'data', 'colormaps')

def load_mat(filename):
    """
    Robustly loads .mat files.
    - First tries scipy.io.loadmat (fast, compatible with v7).
    - If that fails (e.g., v7.3), tries mat73.
    - aggressively filters 'unsupported type' errors if possible.
    """
    data = None
    try:
        # standard load
        data = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except Exception:
        # Fallback to mat73
        try:
            data = mat73.loadmat(filename)
        except Exception as e:
            print(f"Warning: Failed to load {filename} fully. {e}")
            return {}

    def _clean_item(item):
        # Recursively convert structs -> dicts
        if isinstance(item, sio.matlab.mat_struct):
            return {k: _clean_item(getattr(item, k)) for k in item._fieldnames}
        if isinstance(item, dict):
            return {k: _clean_item(v) for k, v in item.items()}
        if isinstance(item, np.ndarray):
            if item.ndim == 0: return item.item()
            if item.size == 1 and np.issubdtype(item.dtype, np.number): return item.item()
        return item

    if isinstance(data, dict):
        # Filter internal keys
        return {k: _clean_item(v) for k, v in data.items() if not k.startswith('__')}
    
    # If mat73 returned a non-dict (rare)
    return _clean_item(data)

def regularize_covariance(sigma, max_cond=500):
    vals, vecs = np.linalg.eigh(sigma)
    
    min_eig = np.min(vals)
    max_eig = np.max(vals)
    
    # Safety floor to match MATLAB's non-negative assumption if numerical noise occurs
    if min_eig <= 0: min_eig = 1e-15
        
    current_cond = max_eig / min_eig
    
    if abs(current_cond) > max_cond:
        lambda_val = (max_eig - max_cond * min_eig) / (max_cond - 1)
        sigma_reg = sigma + lambda_val * np.eye(sigma.shape[0])
    else:
        sigma_reg = sigma
        
    return sigma_reg

def estimate_noise_covariance(data, frame_size=10):
    nx, ny, nt = data.shape
    fs = frame_size if isinstance(frame_size, int) else frame_size[0]
    
    mask = np.ones((nx, ny), dtype=bool)
    mask[fs : nx - fs, fs : ny - fs] = False
    
    bg_voxels = data[mask].reshape(-1, nt, order='F')
    sigma = np.cov(bg_voxels, rowvar=False)
    return regularize_covariance(sigma, 500)

def crop_image(img, target_shape):
    nx, ny = img.shape[:2]
    nnx, nny = target_shape
    
    if nx < nnx:
        xpad = int(round((nnx - nx) / 2))
        pad_width = ((xpad, nnx - nx - xpad), (0,0))
        if img.ndim > 2: pad_width += ((0,0),) * (img.ndim - 2)
        img = np.pad(img, pad_width, mode='constant')
        xrange = slice(0, nnx)
    else:
        start = int(round(nx / 2 - nnx / 2))
        xrange = slice(start, start + nnx)
        
    if ny < nny:
        ypad = int(round((nny - ny) / 2))
        pad_width = ((0,0), (ypad, nny - ny - ypad))
        if img.ndim > 2: pad_width += ((0,0),) * (img.ndim - 2)
        img = np.pad(img, pad_width, mode='constant')
        yrange = slice(0, nny)
    else:
        start = int(round(ny / 2 - nny / 2))
        yrange = slice(start, start + nny)
        
    return img[xrange, yrange, ...]

def extract_roi_stats(roi_masks, ref_map, lrt_res, bayes_res):
    stats = {k: [] for k in ['ref_mean', 'lrt_mean', 'lrt_ci_low', 'lrt_ci_high', 
                             'bayes_mean', 'bayes_ci_low', 'bayes_ci_high']}
    
    if isinstance(roi_masks, dict):
        masks = list(roi_masks.values())
    elif roi_masks.ndim == 3:
        masks = [roi_masks[:,:,i] for i in range(roi_masks.shape[2])]
    else:
        masks = [roi_masks]
        
    for mask_raw in masks:
        m = crop_image(mask_raw > 0, ref_map.shape)
        if np.sum(m) == 0: continue
        
        stats['ref_mean'].append(np.nanmean(ref_map[m]))
        
        # Point Estimates
        stats['lrt_mean'].append(np.nanmean(lrt_res['q'][m]))
        stats['bayes_mean'].append(np.nanmean(bayes_res['q'][m]))
        
        # LRT CI
        stats['lrt_ci_low'].append(np.nanmean(lrt_res['q_ci'][m, 0]))
        stats['lrt_ci_high'].append(np.nanmean(lrt_res['q_ci'][m, 1]))
        
        # Bayes CI
        stats['bayes_ci_low'].append(np.nanmean(bayes_res['q_ci'][m, 0]))
        stats['bayes_ci_high'].append(np.nanmean(bayes_res['q_ci'][m, 1]))
        
    return {k: np.array(v) for k, v in stats.items()}

# --- Colormap Helpers ---
def get_uq_colormap():
    N = 256
    r = np.linspace(0.267, 0.999, N)
    g = np.linspace(0.005, 0.893, N)
    b = np.linspace(0.33, 0.05, N)
    colors = np.stack([r, g, b], axis=1)
    return ListedColormap(colors, name='matlab_uq')

def color_log_remap(ori_cmap, lo, up):
    assert up > 0 and up > lo
    n_map = ori_cmap.shape[0]
    e_inv = np.exp(-1.0)
    a_val = e_inv * up
    m_val = max(a_val, lo)
    b_val = (1.0 / n_map) + (a_val >= lo) * ((a_val - lo) / (2 * a_val - lo)) + 1e-7
    
    log_cmap = np.zeros_like(ori_cmap)
    log_cmap[0, :] = ori_cmap[0, :]
    log_portion = 1.0 / (np.log(m_val) - np.log(up))
    
    for g in range(1, n_map):
        x = (g + 1) * (up - lo) / n_map + lo
        f = 0.0
        if x > m_val:
            f = n_map * ((np.log(m_val) - np.log(x)) * log_portion * (1 - b_val) + b_val)
        else:
            if (lo < a_val) and (x > lo):
                f = n_map * ((x - lo) / (a_val - lo) * (b_val - (1.0 / n_map))) + 1.0
            if x <= lo: f = 1.0
        
        idx = int(min(n_map - 1, np.floor(f)))
        log_cmap[g, :] = ori_cmap[idx, :]
        
    return log_cmap

def relaxation_colormap(maptype, x, lo, up):
    if maptype in ['T1', 'R1']:
        fname = os.path.join(CMAP_DIR, 'lipari.npy')
    else:
        fname = os.path.join(CMAP_DIR, 'navia.npy')
        
    try:
        colortable = np.load(fname)
    except FileNotFoundError:
        colortable = plt.cm.gray(np.linspace(0,1,256))[:,:3]

    if maptype.startswith('R'): colortable = colortable[::-1, :]
    colortable[0, :] = 0 
    eps = (up - lo) / colortable.shape[0]
    x_clip = np.where(x < eps, lo - eps, x)
    lut_cmap = color_log_remap(colortable, lo, up)
    return x_clip, ListedColormap(lut_cmap)

def save_map(data, path, title, rng, mask=None, map_type='T1'):
    if mask is None: mask = np.ones_like(data)
    masked_data = data * mask
    lo, up = rng
    
    if map_type == 'UQ':
        img_disp = np.clip(masked_data, lo, up)
        cmap = get_uq_colormap()
    else:
        img_disp, cmap = relaxation_colormap(map_type, masked_data, lo, up)
    
    plt.figure()
    plt.imshow(img_disp, cmap=cmap, vmin=lo, vmax=up, origin='upper')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

    tiff_path = path.replace('.png', '.tiff')
    plt.imsave(tiff_path, img_disp, cmap=cmap, vmin=lo, vmax=up, format='tiff')
