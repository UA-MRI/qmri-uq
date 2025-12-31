"""
T2 Phantom Validation Script

Loads T2-weighted phantom data and compares mapping methods against a 
Single-Echo Spin-Echo (SESE) reference. Evaluates performance under:
- Free B1 fitting
- Constrained B1 fitting (using external maps)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_mat, estimate_noise_covariance, crop_image, extract_roi_stats, save_map
from solvers import fit_mri_params_lrt, fit_mri_params_bayesian

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'phantom_t2')
DICT_DIR = os.path.join(REPO_ROOT, 'data', 'dictionaries')

USE_IDENTITY_COV = False
FOV = (96, 96)
ALPHA = 0.05
TRUNC_FACTOR = 3.0
RNG_T2 = (0, 150)
RNG_UQ = (0, 40)

OUT_DIR = os.path.join(REPO_ROOT, 'python_output', 'phantom_t2_results_identity' if USE_IDENTITY_COV else 'phantom_t2_results')
os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)

scans = [
    {'name': '8192_NUFFT', 'file': 'scan_8192_nufft.mat', 'dict': 't2_phantom_TE.mat'},
    {'name': '384_LLR',    'file': 'scan_384_llr.mat',    'dict': 't2_phantom_PC.mat'},
    {'name': '192_LLR',    'file': 'scan_192_llr.mat',    'dict': 't2_phantom_PC.mat'}
]
b1_modes = ['Free', 'Range']

def main():
    print("Loading Reference Data...")
    try:
        ref_data = load_mat(os.path.join(DATA_DIR, 'reference_maps.mat'))
        roi_data = load_mat(os.path.join(DATA_DIR, 'roi_masks.mat'))
        
        sese_t2 = crop_image(ref_data['sese_t2map'], FOV)
        sese_con = crop_image(ref_data['sese_contrast'], FOV)
        tfl_b1 = crop_image(ref_data['tfl_b1map'], FOV)
        
        disp_mask = sese_con[:, :, 0] > 0.1 * np.max(sese_con)
    except Exception as e:
        print(f"Error loading references: {e}")
        return

    plot_db = {} 

    for scan in scans:
        print(f"\n--- Processing Scan: {scan['name']} ---")
        try:
            D = load_mat(os.path.join(DICT_DIR, scan['dict']))['D']
            mat_file = load_mat(os.path.join(DATA_DIR, scan['file']))
            contrast = mat_file['contrast']
            
            # --- Strict Header Check ---
            if 'header' not in mat_file:
                raise ValueError(f"File {scan['file']} does not contain 'header'. Cannot determine Echo Spacing.")
            
            header = mat_file['header']
            if 'esp' not in header or 'etl' not in header:
                raise ValueError(f"Header in {scan['file']} missing 'esp' or 'etl'.")
                
            esp = float(header['esp'])
            etl = int(header['etl'])
            
            # Construct TE Array based on File Header
            te_array = np.arange(1, etl + 1) * esp
            
            contrast = crop_image(contrast.astype(np.complex128), FOV)
            
            if USE_IDENTITY_COV:
                sigma = np.eye(contrast.shape[2])
            else:
                sigma = estimate_noise_covariance(contrast, frame_size=10)
                
            for mode in b1_modes:
                print(f"   > Mode: B1 {mode}")
                
                ops = {
                    'alpha': ALPHA,
                    'te_truncation': (contrast.shape[2] == etl),
                    'te_array': te_array,
                    'trunc_factor': TRUNC_FACTOR
                }
                
                if mode == 'Range':
                    ops['b1_mode'] = 'range'
                    # Constrain B1 search to +/- 10% of the measured map
                    b1_stack = np.stack([0.9 * tfl_b1, 1.1 * tfl_b1], axis=-1)
                    ops['b1_input'] = b1_stack
                else:
                    ops['b1_mode'] = 'none'
                
                # --- Solvers ---
                t0 = time.time()
                _, lrt_res = fit_mri_params_lrt(contrast, sigma, D, ops)
                print(f"     LRT: {time.time()-t0:.2f}s")
                
                t0 = time.time()
                _, bayes_res = fit_mri_params_bayesian(contrast, sigma, D, ops)
                print(f"     Bayes: {time.time()-t0:.2f}s")
                
                # --- Reshape Results (order='F') ---
                nx, ny = FOV
                
                lrt_t2 = lrt_res['q'].reshape(nx, ny, order='F')
                bayes_t2 = bayes_res['q'].reshape(nx, ny, order='F')
                
                lrt_ci = lrt_res['q_ci'].reshape(nx, ny, 2, order='F')
                bayes_ci = bayes_res['q_ci'].reshape(nx, ny, 2, order='F')
                
                lrt_uq = lrt_ci[:,:,1] - lrt_ci[:,:,0]
                bayes_uq = bayes_ci[:,:,1] - bayes_ci[:,:,0]

                # --- Save Images ---
                base = f"{scan['name']}_{mode}B1"
                save_map(lrt_t2, os.path.join(OUT_DIR, 'images', f"{base}_LRT_T2.png"), 
                         f"{base.replace('_',' ')} LRT T2", RNG_T2, disp_mask, 'T2')
                save_map(lrt_uq, os.path.join(OUT_DIR, 'images', f"{base}_LRT_Unc.png"), 
                         f"{base.replace('_',' ')} LRT Unc", RNG_UQ, disp_mask, 'UQ')
                save_map(bayes_t2, os.path.join(OUT_DIR, 'images', f"{base}_Bayes_T2.png"), 
                         f"{base.replace('_',' ')} Bayes T2", RNG_T2, disp_mask, 'T2')
                save_map(bayes_uq, os.path.join(OUT_DIR, 'images', f"{base}_Bayes_Unc.png"), 
                         f"{base.replace('_',' ')} Bayes Unc", RNG_UQ, disp_mask, 'UQ')
                         
                # --- Stats ---
                lrt_struct = {'q': lrt_t2, 'q_ci': lrt_ci}
                bayes_struct = {'q': bayes_t2, 'q_ci': bayes_ci}
                
                stats = extract_roi_stats(roi_data['roi_masks'], sese_t2, lrt_struct, bayes_struct)
                plot_db[(scan['name'], mode)] = stats
                
        except Exception as e:
            print(f"Skipping {scan['name']} due to error: {e}")

    generate_plots(plot_db, scans, b1_modes)

def generate_plots(db, scans, modes):
    print("\nGenerating Correlation Plots...")
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    for mode in modes:
        plt.figure(figsize=(10, 5))
        
        for m_idx, method_key in enumerate(['LRT', 'Bayesian']):
            plt.subplot(1, 2, m_idx+1)
            plt.plot([0, 150], [0, 150], 'k--', alpha=0.5)
            
            for s_idx, scan in enumerate(scans):
                key = (scan['name'], mode)
                if key not in db: continue
                
                res = db[key]
                x = res['ref_mean']
                
                if method_key == 'LRT':
                    y = res['lrt_mean']
                    ci_lo = res['lrt_ci_low']
                    ci_hi = res['lrt_ci_high']
                else:
                    y = res['bayes_mean']
                    ci_lo = res['bayes_ci_low']
                    ci_hi = res['bayes_ci_high']
                
                plt.vlines(x, ci_lo, ci_hi, colors=colors[s_idx], alpha=0.5, linewidth=1.5)
                plt.scatter(x, y, color=colors[s_idx], marker=markers[s_idx], 
                            label=scan['name'].replace('_', ' '), zorder=5)
            
            plt.title(f"{method_key} (B1: {mode})")
            plt.xlabel("Reference SESE T2 (ms)")
            plt.ylabel(f"Estimated T2 (ms)")
            plt.xlim(0, 150); plt.ylim(0, 150)
            plt.gca().set_aspect('equal')
            plt.grid(True, alpha=0.3)
            if m_idx == 0: plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'plots', f'Correlation_Plot_{mode}B1.png'))
    
    print(f"Done. Output saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
