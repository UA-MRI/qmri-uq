# solvers.py
import numpy as np
from scipy.stats import chi2

def get_b1_limits(options, dict_b1, n_voxels):
    """
    Determines the B1 search range for each voxel based on the configuration.

    Parameters
    ----------
    options : dict
        - 'b1_mode': 
            'none'  (Search full dictionary range), 
            'map'   (Snap to nearest scalar value in 'b1_input'), 
            'range' (Search within [min, max] defined in 'b1_input').
    dict_b1 : np.ndarray
        Unique B1 values available in the dictionary lookup table.
    n_voxels : int
        Total number of voxels to process.

    Returns
    -------
    np.ndarray
        (n_voxels, 2) array defining the inclusive [min, max] B1 indices/values 
        for each voxel.
    """
    b1_mode = options.get('b1_mode', 'none').lower()
    
    if b1_mode == 'none':
        return np.tile([np.min(dict_b1), np.max(dict_b1)], (n_voxels, 1))
    elif b1_mode == 'map':
        b1_in = options['b1_input'].flatten(order='F')
        idx = np.abs(dict_b1[None, :] - b1_in[:, None]).argmin(axis=1)
        val = dict_b1[idx]
        return np.column_stack((val, val))
    elif b1_mode == 'range':
        b1_in = options['b1_input'].reshape(-1, 2, order='F')
        idx_min = np.abs(dict_b1[None, :] - b1_in[:, 0][:, None]).argmin(axis=1)
        idx_max = np.abs(dict_b1[None, :] - b1_in[:, 1][:, None]).argmin(axis=1)
        return np.column_stack((dict_b1[idx_min], dict_b1[idx_max]))

def fit_mri_params_lrt(data, sigma, D, options=None):
    """
    Unified Likelihood Ratio Test (LRT) solver for qMRI.

    Performs dictionary-based parameter mapping with uncertainty quantification
    using the Profile Likelihood Ratio Test. It analytically minimizes the 
    linear parameters (proton density/phase) to form a profile likelihood 
    dependent only on the non-linear parameters (T1/T2, B1).

    Parameters
    ----------
    data : np.ndarray
        Complex MRI data [Nx, Ny, Nt].
    sigma : np.ndarray
        Noise covariance matrix [Nt, Nt].
    D : dict
        Dictionary structure containing:
        - 'magnetization': [Nt, N_atoms] complex signal traces.
        - 'lookup_table':  [N_atoms, 2] parameter pairs [B1, q].
    options : dict, optional
        - 'alpha': Significance level (default 0.05 for 95% CI).
        - 'te_truncation': Boolean to enable TE-based truncation.
        - 'trunc_factor': Multiplier for T2-based truncation (default 3.0).

    Returns
    -------
    dict, dict
        1. maps: {'q', 'q_mle'} - Point estimates (Cosine Sim and MLE).
        2. stats: {'q_ci'} - Confidence Intervals [Nx*Ny, 2].
    """
    if options is None: options = {}
    alpha = options.get('alpha', 0.05)
    
    # Handle inputs (support both dict and object-like access if needed)
    D_mag = D['magnetization'] if isinstance(D, dict) else D.magnetization
    D_lut = D['lookup_table'] if isinstance(D, dict) else D.lookup_table
    
    nx, ny, nt = data.shape
    n_voxels = nx * ny
    Xobs = data.reshape(n_voxels, nt, order='F').T 

    dict_b1 = np.unique(D_lut[:, 0])
    b1_limits = get_b1_limits(options, dict_b1, n_voxels)
    
    res = {
        'q': np.zeros(n_voxels),      
        'q_mle': np.zeros(n_voxels),
        'q_ci': np.zeros((n_voxels, 2)),
    }

    # Chi-Squared Threshold Determination
    # DoF = 1 if B1 is fixed (Restricted Model), DoF = 2 if B1 is free
    is_fixed = (b1_limits[:, 0] == b1_limits[:, 1])
    dof = np.full(n_voxels, 2)
    dof[is_fixed] = 1
    chi2_vals = chi2.ppf(1 - alpha, dof)

    # Group voxels by B1 constraints to vectorize operations
    unique_ranges = np.unique(b1_limits, axis=0)
    
    for r_min, r_max in unique_ranges:
        v_idx = np.where((b1_limits[:, 0] == r_min) & (b1_limits[:, 1] == r_max))[0]
        if len(v_idx) == 0: continue
        
        # Slice dictionary for this group
        mask = (D_lut[:, 0] >= r_min) & (D_lut[:, 0] <= r_max)
        D_sub = D_mag[:, mask]
        lut_sub = D_lut[mask, :]
        
        # --- 1. Initial Estimate (Cosine Similarity) ---
        X_sub = Xobs[:, v_idx]
        X_norm = X_sub / np.linalg.norm(X_sub, axis=0, keepdims=True)
        X_norm[~np.isfinite(X_norm)] = 0.0
        
        ip = np.abs(X_norm.conj().T @ D_sub)
        best_atom = np.argmax(ip, axis=1)
        q_est = lut_sub[best_atom, 1]
        
        # --- 2. Determine Truncation Lengths ---
        # Used for accelerating T2 mapping by discarding noise-only echoes
        if options.get('te_truncation', False):
            te_array = options['te_array'].flatten()
            trunc_factor = options.get('trunc_factor', 3.0)
            cutoff_times = q_est * trunc_factor
            trunc_lengths = np.sum(te_array[None, :] <= (cutoff_times[:, None] + 1e-9), axis=1)
        else:
            trunc_lengths = np.full(len(v_idx), nt)
            
        # Update point estimates only for valid fits
        valid_trunc = trunc_lengths >= 3
        valid_v_idx = v_idx[valid_trunc]
        if len(valid_v_idx) > 0:
            res['q'][valid_v_idx] = q_est[valid_trunc]
            
        # --- 3. Inner Loop: Process by Truncation Length ---
        u_lengths, group_map = np.unique(trunc_lengths, return_inverse=True)
        
        for len_idx, L in enumerate(u_lengths):
            if L < 3: continue
            
            sub_v_idx_local = np.where(group_map == len_idx)[0]
            final_v_idx = v_idx[sub_v_idx_local]
            
            X_L = Xobs[:L, final_v_idx]
            D_L = D_sub[:L, :]
            Sigma_L = sigma[:L, :L]
            
            # Solve Generalized Linear Model: min || S^-1/2 (X - rho * D) ||^2
            # Use lstsq for stability with ill-conditioned covariance
            S_inv_D = np.linalg.lstsq(Sigma_L, D_L, rcond=None)[0]
            S_inv_X = np.linalg.lstsq(Sigma_L, X_L, rcond=None)[0]
            
            # Precompute terms for Profile Likelihood
            A = np.real(np.sum(X_L.conj() * S_inv_X, axis=0)) # Data Variance
            B = D_L.conj().T @ S_inv_X                        # Cross Correlation
            C = np.real(np.sum(D_L.conj() * S_inv_D, axis=0)) # Model Variance
            
            # Profile Residual: RSS(q) = A - |B|^2 / C
            term2 = (np.abs(B)**2) / C[:, None]
            resid = A[None, :] - term2
            resid[resid <= 0] = np.finfo(float).eps
            
            nll = L * np.log(resid)
            nll[~np.isfinite(nll)] = np.inf
            
            # Maximum Likelihood Estimate
            min_nll = np.min(nll, axis=0)
            mle_idx = np.argmin(nll, axis=0)
            q_mle_vals = lut_sub[mle_idx, 1]
            res['q_mle'][final_v_idx] = q_mle_vals
            
            # LRT Statistic: 2 * (NLL(q) - NLL(mle))
            with np.errstate(invalid='ignore'):
                lrt_stat = 2 * (nll - min_nll)
            
            lrt_stat[~np.isfinite(lrt_stat)] = np.inf
            lrt_stat[lrt_stat < 0] = 0
            
            # Confidence Interval by thresholding
            thresh = chi2_vals[final_v_idx]
            valid_mask = lrt_stat <= thresh[None, :]
            
            q_grid = lut_sub[:, 1]
            q_tiled = np.tile(q_grid[:, None], (1, len(final_v_idx)))
            q_tiled[~valid_mask] = np.nan
            
            ci_min = np.nanmin(q_tiled, axis=0)
            ci_max = np.nanmax(q_tiled, axis=0)
            
            # Fallback for failed CIs (return point estimate)
            failed_fit = np.isnan(ci_min)
            if np.any(failed_fit):
                ci_min[failed_fit] = q_mle_vals[failed_fit]
                ci_max[failed_fit] = q_mle_vals[failed_fit]
            
            res['q_ci'][final_v_idx, 0] = ci_min
            res['q_ci'][final_v_idx, 1] = ci_max

    return {}, res

def fit_mri_params_bayesian(data, sigma, D, options=None):
    """
    Unified Bayesian solver for qMRI.
    
    Calculates marginal posterior distributions P(q|y) by numerically 
    integrating over the dictionary grid.
    
    This function includes logic adapted from code by Selma Metzner (PTB), 2021.
    
    Original License:
    copyright Selma Metzner (PTB) 2021
    This software is licensed under the BSD-like license:
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.
    """
    if options is None: options = {}
    alpha = options.get('alpha', 0.05)
    
    D_mag = D['magnetization'] if isinstance(D, dict) else D.magnetization
    D_lut = D['lookup_table'] if isinstance(D, dict) else D.lookup_table
    
    nx, ny, nt = data.shape
    n_voxels = nx * ny
    Xobs = data.reshape(n_voxels, nt, order='F').T 

    dict_b1 = np.unique(D_lut[:, 0])
    b1_limits = get_b1_limits(options, dict_b1, n_voxels)
    
    res = {
        'q': np.zeros(n_voxels),
        'q_map': np.zeros(n_voxels),
        'q_ci': np.zeros((n_voxels, 2)),
    }
    
    unique_ranges = np.unique(b1_limits, axis=0)
    
    for r_min, r_max in unique_ranges:
        v_idx = np.where((b1_limits[:, 0] == r_min) & (b1_limits[:, 1] == r_max))[0]
        if len(v_idx) == 0: continue

        mask = (D_lut[:, 0] >= r_min) & (D_lut[:, 0] <= r_max)
        lut_sub = D_lut[mask, :]
        D_sub = D_mag[:, mask]
        
        # Sort to ensure grid structure [B1 x q] for integration
        sort_idx = np.lexsort((lut_sub[:, 1], lut_sub[:, 0])) 
        lut_sub = lut_sub[sort_idx]
        D_sub = D_sub[:, sort_idx]
        
        sub_b1_grid = np.unique(lut_sub[:, 0])
        sub_q_grid = np.unique(lut_sub[:, 1])
        
        # --- 1. Initial Estimate ---
        X_sub = Xobs[:, v_idx]
        X_norm = X_sub / np.linalg.norm(X_sub, axis=0, keepdims=True)
        X_norm[~np.isfinite(X_norm)] = 0
        
        ip = np.abs(X_norm.conj().T @ D_sub)
        best_atom = np.argmax(ip, axis=1)
        q_est = lut_sub[best_atom, 1]
        
        # --- 2. Determine Truncation ---
        if options.get('te_truncation', False):
            te_array = options['te_array'].flatten()
            trunc_factor = options.get('trunc_factor', 3.0)
            cutoff_times = q_est * trunc_factor
            trunc_lengths = np.sum(te_array[None, :] <= (cutoff_times[:, None] + 1e-9), axis=1)
        else:
            trunc_lengths = np.full(len(v_idx), nt)
            
        valid_trunc = trunc_lengths >= 3
        valid_v_idx = v_idx[valid_trunc]
        if len(valid_v_idx) > 0:
            res['q'][valid_v_idx] = q_est[valid_trunc]
            
        u_lengths, group_map = np.unique(trunc_lengths, return_inverse=True)
        
        for len_idx, L in enumerate(u_lengths):
            if L < 3: continue
            
            sub_v_idx_local = np.where(group_map == len_idx)[0]
            final_v_idx = v_idx[sub_v_idx_local]
            
            X_L = Xobs[:L, final_v_idx]
            D_L = D_sub[:L, :]
            Sigma_L = sigma[:L, :L]
            
            # --- Covariance Weighted Projection ---
            S_inv_D = np.linalg.lstsq(Sigma_L, D_L, rcond=None)[0]
            S_inv_X = np.linalg.lstsq(Sigma_L, X_L, rcond=None)[0]
            
            A = np.real(np.sum(X_L.conj() * S_inv_X, axis=0))
            B = D_L.conj().T @ S_inv_X 
            C = np.real(np.sum(D_L.conj() * S_inv_D, axis=0))
            
            term = (np.abs(B)**2) / C[:, None]
            RSS = A[None, :] - term
            RSS[RSS <= 0] = np.finfo(float).eps
            
            # Log Likelihood -> Log Probability (Flat Prior assumed)
            log_prob = (1 - L) * np.log(RSS) - np.log(C[:, None])
            log_prob[~np.isfinite(log_prob)] = -np.inf
            
            # Reshape for Integration [n_q, n_b1, n_vox]
            nq, nb1 = len(sub_q_grid), len(sub_b1_grid)
            log_prob_grid = log_prob.reshape(nq, nb1, len(final_v_idx), order='F')
            
            # Softmax normalization (Log-Sum-Exp trick)
            max_lp = np.nanmax(log_prob.reshape(-1, len(final_v_idx)), axis=0)
            max_lp[~np.isfinite(max_lp)] = 0
            
            prob_grid = np.exp(log_prob_grid - max_lp[None, None, :])
            
            # --- Marginalize over B1 ---
            if nb1 > 1:
                p_q_raw = np.trapz(prob_grid, sub_b1_grid, axis=1)
            else:
                p_q_raw = prob_grid[:, 0, :]
            
            # Normalize PDF
            norm_q = np.trapz(p_q_raw, sub_q_grid, axis=0)
            norm_q[norm_q == 0] = 1.0
            p_q = p_q_raw / norm_q[None, :]
            
            # --- Extract Stats ---
            for i, vid in enumerate(final_v_idx):
                map_idx = np.argmax(p_q[:, i])
                map_val = sub_q_grid[map_idx]
                res['q_map'][vid] = map_val
                lb, ub = calc_ci_greedy(sub_q_grid, p_q[:, i], map_val, 1-alpha)
                res['q_ci'][vid, :] = [lb, ub]
            
    return {}, res

def calc_ci_greedy(grid, prob, center, conf):
    """
    Calculates the Bayesian Credible Interval by expanding greedily
    from the MAP estimate until the cumulative mass reaches 'conf'.
    """
    idx = (np.abs(grid - center)).argmin()
    L, R = idx, idx
    acc_prob = 0.0
    n = len(grid)
    while acc_prob < conf and (L > 0 or R < n-1):
        mass_L = -1
        if L > 0: mass_L = np.trapz(prob[L-1:R+1], grid[L-1:R+1])
        mass_R = -1
        if R < n-1: mass_R = np.trapz(prob[L:R+2], grid[L:R+2])
        if mass_L > mass_R: L -= 1; acc_prob = mass_L
        else: R += 1; acc_prob = mass_R
    return grid[L], grid[R]
