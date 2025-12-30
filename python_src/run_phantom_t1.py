# run_phantom_t1.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import load_mat, estimate_noise_covariance, crop_image, extract_roi_stats, save_map
from solvers import fit_mri_params_lrt, fit_mri_params_bayesian

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'phantom_t1')
DICT_DIR = os.path.join(REPO_ROOT, 'data', 'dictionaries')

USE_IDENTITY_COV = False
FOV = (96, 96)
ALPHA = 0.05
RNG_T1 = (0, 1500)
RNG_UQ = (0, 200)

OUT_DIR = os.path.join(REPO_ROOT, 'python_output', 'phantom_t1_results_identity' if USE_IDENTITY_COV else 'phantom_t1_results')
os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)

experiments = [
    {
        'name': '1Meas_LLR', 'type': 'PC',
        'dicts': ['t1_phantom_PC_50TI.mat', 't1_phantom_PC_30TI.mat', 't1_phantom_PC_10TI.mat'],
        'tis': [50, 30, 10],
        'files': ['contrast_1meas_llr_50ti.mat', 'contrast_1meas_llr_30ti.mat', 'contrast_1meas_llr_10ti.mat']
    },
    {
        'name': '16Meas_NUFFT', 'type': 'TI',
        'dicts': ['t1_phantom_TI_50TI.mat', 't1_phantom_TI_30TI.mat', 't1_phantom_TI_10TI.mat'],
        'tis': [50, 30, 10],
        'files': ['contrast_16meas_nufft_50ti.mat', 'contrast_16meas_nufft_30ti.mat', 'contrast_16meas_nufft_10ti.mat']
    }
]

def main():
    print("Loading Reference Data...")
    try:
        ref_data = load_mat(os.path.join(DATA_DIR, 'reference_maps.mat'))
        roi_data = load_mat(os.path.join(DATA_DIR, 'roi_masks.mat'))
        
        seir_t1 = crop_image(ref_data['seir_t1map'], FOV)
        seir_con = crop_image(ref_data['seir_contrast'], FOV)
        
        disp_mask = seir_con[:, :, -1] > 0.1 * np.max(seir_con)
    except Exception as e:
        print(f"Error loading references: {e}")
        return

    plot_db = {} 

    for exp in experiments:
        print(f"\n=== Experiment: {exp['name']} ({exp['type']} Space) ===")
        
        for i, n_ti in enumerate(exp['tis']):
            fname = exp['files'][i]
            dname = exp['dicts'][i]
            
            print(f"   > Truncation: {n_ti} TIs ({fname})")
            
            try:
                D = load_mat(os.path.join(DICT_DIR, dname))['D']
                contrast = load_mat(os.path.join(DATA_DIR, fname))['contrast']
                
                # Cast to complex128 to preserve phase information
                contrast = crop_image(contrast.astype(np.complex128), FOV) 
                
                if USE_IDENTITY_COV:
                    sigma = np.eye(contrast.shape[2])
                else:
                    sigma = estimate_noise_covariance(contrast, frame_size=10)
                
                ops = {'alpha': ALPHA, 'b1_mode': 'none', 'te_truncation': False}
                
                # --- Solvers ---
                t0 = time.time()
                _, lrt_res = fit_mri_params_lrt(contrast, sigma, D, ops)
                print(f"     LRT: {time.time()-t0:.2f}s")
                
                t0 = time.time()
                _, bayes_res = fit_mri_params_bayesian(contrast, sigma, D, ops)
                print(f"     Bayes: {time.time()-t0:.2f}s")
                
                # --- Reshape Results (order='F') ---
                nx, ny = FOV
                
                # Reshape Point Estimates
                lrt_t1 = lrt_res['q'].reshape(nx, ny, order='F')
                bayes_t1 = bayes_res['q'].reshape(nx, ny, order='F')
                
                # Reshape CIs
                lrt_ci = lrt_res['q_ci'].reshape(nx, ny, 2, order='F')
                bayes_ci = bayes_res['q_ci'].reshape(nx, ny, 2, order='F')
                
                lrt_uq_img = lrt_ci[:,:,1] - lrt_ci[:,:,0]
                bayes_uq_img = bayes_ci[:,:,1] - bayes_ci[:,:,0]
                
                # --- Save Images ---
                base = f"{exp['name']}_{n_ti}TIs"
                save_map(lrt_t1, os.path.join(OUT_DIR, 'images', f"{base}_LRT_T1.png"), 
                         f"{base.replace('_',' ')} LRT T1", RNG_T1, disp_mask, 'T1')
                save_map(lrt_uq_img, os.path.join(OUT_DIR, 'images', f"{base}_LRT_Unc.png"), 
                         f"{base.replace('_',' ')} LRT Unc", RNG_UQ, disp_mask, 'UQ')
                save_map(bayes_t1, os.path.join(OUT_DIR, 'images', f"{base}_Bayes_T1.png"), 
                         f"{base.replace('_',' ')} Bayes T1", RNG_T1, disp_mask, 'T1')
                save_map(bayes_uq_img, os.path.join(OUT_DIR, 'images', f"{base}_Bayes_Unc.png"), 
                         f"{base.replace('_',' ')} Bayes Unc", RNG_UQ, disp_mask, 'UQ')

                # --- Extract Stats ---
                # Pass correct shapes to extractor
                lrt_struct = {'q': lrt_t1, 'q_ci': lrt_ci}
                bayes_struct = {'q': bayes_t1, 'q_ci': bayes_ci}
                
                stats = extract_roi_stats(roi_data['roi_masks'], seir_t1, lrt_struct, bayes_struct)
                plot_db[(exp['name'], n_ti)] = stats
                
            except Exception as e:
                print(f"     Failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    generate_plots(plot_db, experiments)

def generate_plots(db, experiments):
    print("\nGenerating Correlation Plots...")
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']
    
    # Create one Figure per Experiment
    for exp in experiments:
        fig = plt.figure(figsize=(12, 6))
        methods = ['LRT', 'Bayesian']
        
        for m_idx, method_key in enumerate(methods):
            plt.subplot(1, 2, m_idx+1)
            plt.plot([0, 2000], [0, 2000], 'k--', alpha=0.5)
            
            for t_i, n_ti in enumerate(exp['tis']):
                key = (exp['name'], n_ti)
                if key not in db: continue
                
                res = db[key]
                x = res['ref_mean']
                
                # Switch Data Source
                if method_key == 'LRT':
                    y = res['lrt_mean']
                    ci_lo = res['lrt_ci_low']
                    ci_hi = res['lrt_ci_high']
                else:
                    y = res['bayes_mean']
                    ci_lo = res['bayes_ci_low']
                    ci_hi = res['bayes_ci_high']
                
                # Plot Errors and Points
                plt.vlines(x, ci_lo, ci_hi, colors=colors[t_i], alpha=0.5, linewidth=1.5)
                plt.scatter(x, y, color=colors[t_i], marker=markers[t_i], 
                            label=f"{n_ti} TIs", zorder=5)
            
            plt.title(f"{exp['name'].replace('_', ' ')}: {method_key}")
            plt.xlabel("Reference SEIR T1 (ms)")
            plt.ylabel(f"Estimated T1 (ms)")
            plt.xlim(400, 1600)
            plt.ylim(400, 1600)
            plt.gca().set_aspect('equal')
            plt.grid(True, alpha=0.3)
            if m_idx == 0: plt.legend(loc='upper left')

        plt.tight_layout()
        save_name = os.path.join(OUT_DIR, 'plots', f"Correlation_{exp['name']}.png")
        plt.savefig(save_name)
        print(f"Saved {save_name}")

    print(f"Done. Output saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
