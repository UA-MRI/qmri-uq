"""
Numerical Simulation Runner

Performs Monte Carlo simulations to validate the statistical methods.
- Generates synthetic signals for TE (T2) and PC (Subspace) contrast.
- Adds correlated complex Gaussian noise (colored noise).
- Compares LRT and Bayesian coverage probabilities against the ground truth.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from utils import load_mat, regularize_covariance, estimate_noise_covariance
from solvers import fit_mri_params_lrt, fit_mri_params_bayesian

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'simulation_phantom')
DICT_DIR = os.path.join(REPO_ROOT, 'data', 'dictionaries')

SNR_DB = 15
T2_ARRAY = np.arange(20, 301, 20) 
B1_SIM = 1.0
N_SIM = 100
ALPHA = 0.05
USE_IDENTITY_COV = False

if USE_IDENTITY_COV:
    OUT_DIR = os.path.join(REPO_ROOT, 'python_output', 'simulation_results_identity')
else:
    OUT_DIR = os.path.join(REPO_ROOT, 'python_output', 'simulation_results')
    
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    np.random.seed(2025)
    contrast_modes = ['TE', 'PC']
    b1_modes = [True, False]
    plot_data = []

    for contrast_type in contrast_modes:
        print(f"\n=== Processing {contrast_type} ===")
        try:
            h_data = load_mat(os.path.join(DATA_DIR, 'header.mat'))
            c_data = load_mat(os.path.join(DATA_DIR, 'contrast.mat'))
            d_data = load_mat(os.path.join(DICT_DIR, f'simulation_{contrast_type}.mat'))
            
            contrast = c_data['contrast'] 
            D = d_data['D']
            
            if 'header' not in h_data:
                raise ValueError("header.mat missing 'header' struct.")
            header = h_data['header']
            
        except Exception as e:
            print(f"Error loading files: {e}")
            continue

        if contrast_type == 'PC':
            # Project onto subspace basis (u)
            basis = D['u']
            nx, ny, n_orig = contrast.shape
            c_flat = contrast.reshape(nx*ny, n_orig, order='F')
            c_proj = c_flat @ basis
            contrast = c_proj.reshape(nx, ny, -1, order='F')
            N_t = int(basis.shape[1])
        else:
            N_t = int(header['etl'])

        # Prepare Truncation Array (TE Only)
        if contrast_type == 'TE':
            if 'esp' not in header:
                raise ValueError("Header missing 'esp' for TE simulation.")
            TE_array_full = np.arange(1, header['etl'] + 1) * float(header['esp'])
        else:
            TE_array_full = None

        # --- Noise Coloring ---
        # Scale noise covariance to match simulation SNR
        contrast_dbl = contrast.astype(np.complex128) * 1e4
        sigma_bg = estimate_noise_covariance(contrast_dbl, frame_size=10)
        
        sig_norm = 1.0
        snr_lin = 10**(SNR_DB / 10.0)
        cov_scale = sig_norm**2 / (snr_lin**2 * np.trace(sigma_bg).real)
        sigma_sim = regularize_covariance(cov_scale * sigma_bg, 500)
        
        # Block matrix for Complex Gaussian noise
        Sigma_w = np.block([
            [np.real(sigma_sim), -np.imag(sigma_sim)],
            [np.imag(sigma_sim),  np.real(sigma_sim)]
        ])
        
        sigma_fit = np.eye(N_t) if USE_IDENTITY_COV else sigma_sim
            
        dict_atoms = D['magnetization']
        dict_lut = D['lookup_table']
        
        for use_b1_map in b1_modes:
            if use_b1_map:
                lbl = f"{contrast_type} Constrained B1"
                b1_mode_str = 'range'
                b1_input_range = np.array([0.9, 1.1])
                csv_tag = 'ConstrainedB1'
                color = 'r' if contrast_type == 'TE' else 'b'
            else:
                lbl = f"{contrast_type} Free B1"
                b1_mode_str = 'none'
                b1_input_range = None
                csv_tag = 'FreeB1'
                color = 'k' if contrast_type == 'TE' else 'm'
                
            print(f"  > Running {lbl}...")
            res_struct = {'T2': [], 'LRT_Cov': [], 'Bayes_Cov': [], 'LRT_Size': [], 'Bayes_Size': []}
            
            for t2_true in tqdm(T2_ARRAY):
                dist = np.abs(dict_lut[:,0] - B1_SIM) + np.abs(dict_lut[:,1] - t2_true)
                idx = np.argmin(dist)
                clean_sig = dict_atoms[:, idx]
                clean_sig = clean_sig / np.linalg.norm(clean_sig) * sig_norm
                
                # Add Noise
                noise_ri = np.random.multivariate_normal(np.zeros(2*N_t), Sigma_w, N_SIM)
                noise_c = noise_ri[:, :N_t] + 1j * noise_ri[:, N_t:]
                noisy_sigs = (clean_sig[None, :] + noise_c).reshape(N_sim_img(N_SIM), order='F')
                
                ops = {'alpha': ALPHA, 'b1_mode': b1_mode_str}
                if use_b1_map:
                    ops['b1_input'] = np.tile(b1_input_range, (N_SIM, 1, 1))
                
                if contrast_type == 'TE':
                    ops['te_truncation'] = True
                    ops['te_array'] = TE_array_full
                    ops['trunc_factor'] = 3.0
                else:
                    ops['te_truncation'] = False

                _, lrt_res = fit_mri_params_lrt(noisy_sigs, sigma_fit, D, ops)
                _, bayes_res = fit_mri_params_bayesian(noisy_sigs, sigma_fit, D, ops)
                
                l_ci = lrt_res['q_ci']
                b_ci = bayes_res['q_ci']
                
                l_cov = np.mean((t2_true >= l_ci[:,0]) & (t2_true <= l_ci[:,1])) * 100
                b_cov = np.mean((t2_true >= b_ci[:,0]) & (t2_true <= b_ci[:,1])) * 100
                l_sz = np.nanmean(l_ci[:,1] - l_ci[:,0])
                b_sz = np.nanmean(b_ci[:,1] - b_ci[:,0])
                
                res_struct['T2'].append(t2_true)
                res_struct['LRT_Cov'].append(l_cov)
                res_struct['Bayes_Cov'].append(b_cov)
                res_struct['LRT_Size'].append(l_sz)
                res_struct['Bayes_Size'].append(b_sz)
                
            plot_data.append({'label': lbl, 'color': color, 'res': res_struct})
            
            csv_path = os.path.join(OUT_DIR, f'sim_results_{SNR_DB}dB_{contrast_type}_{csv_tag}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['T2', 'LRT_Cov', 'Bayes_Cov', 'LRT_Size', 'Bayes_Size'])
                rows = zip(res_struct['T2'], res_struct['LRT_Cov'], res_struct['Bayes_Cov'], 
                           res_struct['LRT_Size'], res_struct['Bayes_Size'])
                writer.writerows(rows)

    plot_results(plot_data, OUT_DIR)

def N_sim_img(n_sim): return (n_sim, 1, -1)

def plot_results(data, out_dir):
    plt.figure(figsize=(10, 6))
    plt.axhline(95, color='gray', linestyle='--')
    for d in data:
        r = d['res']
        plt.plot(r['T2'], r['LRT_Cov'], linestyle='--', color=d['color'], label=f"LRT {d['label']}")
        plt.plot(r['T2'], r['Bayes_Cov'], linestyle=':', color=d['color'], label=f"Bayes {d['label']}")
    plt.ylim(70, 101)
    plt.xlabel('T2 (ms)')
    plt.ylabel('Coverage (%)')
    plt.legend(ncol=2)
    plt.savefig(os.path.join(out_dir, 'MASTER_coverage.png'))
    
    plt.figure(figsize=(10, 6))
    for d in data:
        r = d['res']
        plt.plot(r['T2'], r['LRT_Size'], linestyle='--', color=d['color'], label=f"LRT {d['label']}")
        plt.plot(r['T2'], r['Bayes_Size'], linestyle=':', color=d['color'], label=f"Bayes {d['label']}")
    plt.xlabel('T2 (ms)')
    plt.ylabel('Interval Size (ms)')
    plt.legend(ncol=2)
    plt.savefig(os.path.join(out_dir, 'MASTER_size.png'))

if __name__ == "__main__":
    main()
