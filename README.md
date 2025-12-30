# Dictionary-Based MRI Uncertainty Quantification

This repository contains the MATLAB implementation for **Uncertainty Quantification (UQ) in Dictionary-Based Quantitative MRI**. It provides unified solvers for Likelihood Ratio Test (LRT) and Bayesian methods to estimate parameter confidence intervals (CIs) alongside standard quantitative maps.

The code reproduces the numerical simulations and phantom validation experiments described in the associated paper.

## Repository Structure

The repository is organized as follows:

```text
.
├── data/                       # Input data (must be populated with .mat files)
│   ├── simulation_phantom/     # Data for numerical simulations (header, contrast)
│   ├── phantom_t1/             # T1 phantom scans (LLR/NUFFT) and reference maps
│   ├── phantom_t2/             # T2 phantom scans (8192/384/192 views) and references
│   └── dictionaries/           # Pre-computed dictionaries (TE/PC space, T1/T2)
│
├── matlab_src/                 # Source code
│   ├── run_simulation.m        # Main script for Numerical Simulations (Fig. 2)
│   ├── run_phantom_t1.m        # Main script for T1 Phantom Validation
│   ├── run_phantom_t2.m        # Main script for T2 Phantom Validation
│   │
│   ├── utils/                  # Core solvers and helper functions
│   │   ├── fit_mri_params_lrt.m        # Unified Frequentist (LRT) Solver
│   │   ├── fit_mri_params_bayesian.m   # Unified Bayesian Solver
│   │   ├── estimateNoiseCovariance.m   # Noise covariance estimation
│   │   ├── save_t1_img.m / save_t2_img.m # Image export helpers
│   │   └── ...
│   │
│   └── qmri_cmaps/             # Colormap utilities
│       └── relaxationColorMap.m
│
├── matlab_output/                     # MATLAB Generated results (created automatically)
│   ├── simulation_results/
│   ├── phantom_t1_results/
│   └── phantom_t2_results/
│
└── python_output/                     # python Generated results (created automatically)
    ├── simulation_results/
    ├── phantom_t1_results/
    └── phantom_t2_results/

```

## Setup & Requirements

1. **MATLAB:** The code was developed and tested in MATLAB R2020a.
2. **Pathing:** The scripts automatically add the `utils` and `qmri_cmaps` folders to the path relative to `matlab_src`. Ensure you run the scripts from within the `matlab_src` folder or keep the folder structure intact.
3. **Data:** Ensure the `data/` directory contains the required `.mat` files (dictionaries, contrast images, and headers) before running experiments.

---

## Usage

### 1. Numerical Simulations (`run_simulation.m`)

Reproduces the Monte Carlo simulations (Figure 2 in the paper) to evaluate CI coverage and interval width under controlled noise.

* **Scenarios:** Loops through **TE vs. PC** subspace contrasts and **Free vs. Restricted** B1 mapping.
* **Configuration:**
* `SNR_dB`: Signal-to-Noise Ratio (default: 15 dB).
* `N_sim`: Number of Monte Carlo realizations (default: 1000).
* `use_identity_cov`: Set to `true` to test the "i.i.d. noise" failure mode.


* **Output:** Saves CSV statistics and combined "Master" plots (Coverage & Size) to `output/simulation_results/`.

### 2. T2 Phantom Validation (`run_phantom_t2.m`)

Validates T2 mapping uncertainty on phantom data across different acceleration factors.

* **Scenarios:**
* **Acquisitions:** 8192-view (NUFFT Reference), 384-view (LLR), 192-view (LLR).
* **B1 Constraints:** "Free" (Joint estimation) vs. "Range" (Restricted to ±10% of measured B1).


* **Output:**
* **Images:** T2 maps and Uncertainty maps (PNG/TIFF) in `output/phantom_t2_results/images/`.
* **Plots:** Square correlation plots comparing estimated T2 vs. Gold Standard SESE T2.



### 3. T1 Phantom Validation (`run_phantom_t1.m`)

Validates T1 mapping uncertainty using Inversion Recovery (IR) sequences with retrospective acceleration.

* **Scenarios:**
* **1-Meas LLR:** Single measurement reconstructed in Principal Component (PC) space.
* **16-Meas NUFFT:** 16 averages reconstructed in Time (TI) space.
* **Truncation:** Retrospective acceleration by using only the first 50, 30, or 10 TIs.


* **Output:**
* **Images:** T1 maps and Uncertainty maps in `output/phantom_t1_results/images/`.
* **Plots:** Correlation plots comparing Estimated T1 vs. Reference SEIR T1.



---

## Key Configuration Options

All three scripts include a standard configuration block at the top:

* **`use_identity_cov` (boolean):**
* `false` (Default): Estimates the noise covariance matrix empirically from the data background.
* `true`: Forces the use of an Identity matrix. This is used to demonstrate failure modes when noise correlations are ignored.
* *Note:* When `true`, output folders are suffixed with `_identity` (e.g., `phantom_t2_results_identity`) to prevent overwriting valid results.


* **`alpha_lvl`:** Significance level for Confidence Intervals (default `0.05` for 95% CIs).

## Core Solvers (`matlab_src/utils/`)

The repository relies on two unified solver functions that handle both T1 and T2 mapping:

1. **`fit_mri_params_lrt(data, sigma, D, options)`**:
* Implements the **Log-Likelihood Ratio Test**.
* Returns Maximum Likelihood Estimates (MLE) and profile likelihood-based Confidence Intervals.


2. **`fit_mri_params_bayesian(data, sigma, D, options)`**:
* Implements **Bayesian Inversion** with numerical integration.
* Returns Marginal Posterior means (Point Estimates) and Credible Intervals.



Both solvers accept a dictionary structure `D` containing signal atoms and a lookup table, making them agnostic to the specific physics (T1 vs. T2) being modeled.



## Python Implementation

We provide a Python translation of the simulation pipeline to ensure cross-platform reproducibility. There may be some small discrepancies in results due to implementation differences, but overall the two versions provide similar results

### 1. Prerequisites
Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 2. Setup Environment
Navigate to the repository root and create the environment:

```bash
conda env create -f environment.yml
conda activate mri_uq

3. Running the Simulation

To run the T2 mapping numerical simulation:
Bash

python run_simulation.py

Results will be saved in python_output/. The script automatically detects the data in the existing data/ folder (assuming you have already downloaded or generated the MATLAB data files).