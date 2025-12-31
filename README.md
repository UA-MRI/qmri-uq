# Dictionary-Based MRI Uncertainty Quantification

This repository contains the MATLAB implementation for **Uncertainty Quantification (UQ) in Dictionary-Based Quantitative MRI**. It provides unified solvers for Likelihood Ratio Test (LRT) and Bayesian methods to estimate parameter confidence intervals (CIs) alongside standard quantitative maps.

The code reproduces the numerical simulations and phantom validation experiments presented at the ISMRM Workshop on Data Sampling and Image Reconstruction in January, 2026. A longer form journal article is in preparation. 

If you are reading this, the journal article has not been published yet. Keep an eye out for it, but in the meantime, please cite the following abstract:
```bibtex
@inproceedings{ISMRM_Sedona_2026,
  author    = {Toner, Brian and Goerke, Ute and Ahanonu, Eze and Johnson, Kevin and Deshpande, Vibhas and Wu, Holden H and Altbach, M and Bilgin, A},
  title     = {Methods for Uncertainty Quantification in Dictionary Matching to Advance Interpretable Quantitative MRI},
  booktitle = {Proceedings of the ISMRM Workshop on Data Sampling and Image Reconstruction},
  year      = {2026},
  month     = {Jan.},
  days      = {11--14},
  address   = {Sedona, AZ, USA},
  publisher = {International Society for Magnetic Resonance in Medicine (ISMRM)},
  note      = {Abstract \#00187}
}
```

1. B. Toner, U. Goerke, E. Ahanonu, K. Johnson, V. Deshpande, H. H. Wu, M. Altbach, and A. Bilgin, "Methods for uncertainty quantification in dictionary matching to advance interpretable quantitative MRI," in *Proc. ISMRM Workshop Data Sampling Image Reconstr.*, Sedona, AZ, USA, Jan. 11–14, 2026, Abstract 00187.

## Repository Structure

The repository is organized as follows:

```text
.
├── data/                       # Input data (must be populated with .mat files)
│   ├── simulation_phantom/     # Data for numerical simulations (header, contrast)
│   ├── colormaps/     		# saved custom colormaps
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

├── python_src/                 # Python implementation
│   ├── run_simulation.py       # Main script for Numerical Simulations
│   ├── run_phantom_t1.py       # Main script for T1 Phantom Validation
│   ├── run_phantom_t2.py       # Main script for T2 Phantom Validation
│   ├── solvers.py              # Unified LRT and Bayesian solvers
│   └── utils.py                # Helper functions (IO, stats, plotting) ? [Confirm .py or / folder]
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

We provide a Python translation of the simulation and phantom validation pipelines to ensure cross-platform reproducibility.

### Repository Structure (Python)

The Python source code is organized as follows:

```text
├── python_src/                 # Python implementation
│   ├── run_simulation.py       # Main script for Numerical Simulations
│   ├── run_phantom_t1.py       # Main script for T1 Phantom Validation
│   ├── run_phantom_t2.py       # Main script for T2 Phantom Validation
│   ├── solvers.py              # Unified LRT and Bayesian solvers
│   └── utils.py                # Helper functions (IO, stats, plotting)

```

### 1. Prerequisites

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or an equivalent Python environment installed.

### 2. Setup Environment

Navigate to the repository root and create the environment:

```bash
conda env create -f environment.yml
conda activate mri_uq

```

### 3. Usage

All scripts should be run from the `python_src` directory or ensure that directory is in your PYTHONPATH.

#### **Numerical Simulations**

Reproduces the Monte Carlo simulations (similar to Figure 2).

```bash
python python_src/run_simulation.py

```

* **Output:** Statistics and plots saved to `python_output/simulation_results/`.

#### **T2 Phantom Validation**

Runs the validation on T2 phantom data across 8192, 384, and 192 views.

```bash
python python_src/run_phantom_t2.py

```

* **Output:** T2 maps, Uncertainty maps, and correlation plots saved to `python_output/phantom_t2_results/`.

#### **T1 Phantom Validation**

Runs the validation on T1 phantom data (1-Meas LLR vs 16-Meas NUFFT) with retrospective acceleration.

```bash
python python_src/run_phantom_t1.py

```

* **Output:** T1 maps, Uncertainty maps, and correlation plots saved to `python_output/phantom_t1_results/`.

