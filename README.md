# frb-ffpe

**Fractional Fokker–Planck Framework for Fast Radio Burst Statistics**

Ran Duan  
National Astronomical Observatories, Chinese Academy of Sciences  
20 DaTun Road, Chaoyang District, Beijing 100025, China  
Email: duanran@nao.cas.cn

---

## Overview

This repository contains the source code, analysis scripts, and paper source files for:

> **A Fractional Fokker–Planck Framework Unifying Energy and Temporal Statistics of Fast Radio Bursts**

The framework proposes a minimal phenomenological model based on a fractional nonlinear Fokker–Planck equation (FFPE) coupled with a continuous-time random walk (CTRW), capturing:

- **Power-law energy tails** (α ≈ 2.0–2.1)
- **Non-Poissonian Weibull waiting times** (k < 1)
- **q-Gaussian energy-difference distributions** (as an independent consistency check)

with only three physical parameters (κ, σ, γ).

## Repository Structure

```
├── main.tex                        # Main paper source (LaTeX)
├── methods.tex                     # Methods section
├── extended_data.tex               # Extended Data figures and tables
├── references.bib                  # Bibliography
├── compiled.pdf                    # Compiled paper PDF
│
├── generate_figures.py             # Script to generate Figures 1–4
├── model_comparison.py             # Model comparison analysis
├── chime_analysis.py               # CHIME/FRB Catalog 2 cross-validation
│
├── fig1_observational_validation.* # Figure 1: FAST burst validation
├── fig2_mcmc_constraints.*         # Figure 2: MCMC parameter constraints
├── fig3_corner_plots.*             # Figure 3: Corner plots
├── fig4_gamma_age.*                # Figure 4: γ–age relation
├── chime_validation.*              # CHIME cross-validation figure
├── model_comparison.*              # Model comparison figure
│
└── README.md                       # This file
```

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib
- emcee (for MCMC)
- corner (for posterior visualization)

Install dependencies:

```bash
pip install numpy scipy matplotlib emcee corner
```

## Usage

### Generate Main Figures

```bash
python generate_figures.py
```

This produces Figures 1–4 and the Extended Data figures.

### Run Model Comparison

```bash
python model_comparison.py
```

Compares the FFPE framework against four alternative models (SOC, Weibull, q-Gaussian, Memory).

### CHIME Cross-Validation

```bash
python chime_analysis.py
```

Performs cross-validation against CHIME/FRB Catalog 2 data (requires CHIME data files in `../CHIME_data/`).

## Key Results

| Observable | FAST Data | FFPE Prediction | Δ |
|------------|-----------|-----------------|---|
| α (power-law) | 2.03–2.13 | 2.01–2.10 | < 0.05 |
| k (Weibull) | 0.39–0.64 | 0.39–0.62 | < 0.025 |
| q (Tsallis) | 1.63 | 1.52–1.72 | consistent |

### CHIME Cross-Validation

- FRB 20220912A: γ_CHIME = 0.452 ± 0.017 (consistent with FRB 121102 γ = 0.460 at 0.1σ)
- Cross-source α consistency: σ_α = 0.17

## Citation

If you use this code, please cite:

```bibtex
@article{duan2025frb,
  title={A Fractional Fokker--Planck Framework Unifying Energy and Temporal Statistics of Fast Radio Bursts},
  author={Duan, Ran},
  journal={submitted},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Data Availability

- FAST FRB 121102: VizieR (J/Nature/598/267)
- FAST FRB 20201124A: Figshare (doi:10.6084/m9.figshare.20432920)
- CHIME/FRB Catalog 2: CHIME/FRB Open Data Release
