# McCormick Envelope-Guided Adaptive Sampling for Surrogate Modelling

> **GitHub Repository:** [https://github.com/qzramiz/AdaptiveSamplingForSurrogateTraining-McCormickEnvelopes]

---

## Overview

This project investigates **McCormick relaxation-guided adaptive sampling** strategies for building accurate surrogate models of expensive 1D and 2D benchmark functions, as well as a PDE-based heat diffusion simulation. The core idea is to use McCormick envelope gaps—computed via interval arithmetic—to identify regions of high complexity and allocate sampling budget accordingly.

Four families of experiments are covered:

- **Non-hybrid methods** — Pure sampling strategies (Sobol, LHS, Grid, Approach A & B) evaluated once with a fixed budget, followed by GP and MLP surrogate fitting.
- **Hybrid (Bayesian) methods** — An envelope-guided warm start followed by iterative Bayesian active learning using Expected Improvement (EI) or uncertainty (STD) acquisition functions, optionally weighted by domain gap scores. Experiments sweep over domain split count *N* and refinement budget *M*.
- **2D extension** — The McCormick framework is extended to two-dimensional domains and tested on the Langermann and Schwefel benchmark functions.
- **Heat diffusion PDE surrogate** — A data-driven surrogate (MLP and GP) is trained to emulate a 1D transient heat diffusion solver (FTCS scheme), with adaptive temporal snapshot allocation guided by McCormick envelope gaps over the spatial temperature profile.

---

## Repository Structure

```
.
├── functions/
│   └── function.py                        # 1D benchmark functions (Forrester, Schwefel, ...)
├── samplers/
│   └── adaptive_sampler.py                # AdaptiveSampler: envelope-guided point selection
├── interval_builder.py                    # IntervalBuilder: McCormick interval construction
├── envelope_builder.py                    # piecewise_envelopes: piecewise convex/concave hulls
│
├── Non-hybrid_Methods.ipynb               # Experiment 1: baseline sampling strategies
├── Hybrid_Methods-Choice_of_N.ipynb       # Experiment 2: effect of domain splits N
├── Hybrid_Methods_-_Choice_of_M.ipynb     # Experiment 3: effect of refinement budget M
├── Langermann-2D_-_latest.ipynb           # Experiment 4: 2D Langermann function
├── Schwefel-2D.ipynb                      # Experiment 5: 2D Schwefel function
├── HeatDiffusion_-_OG.ipynb               # Experiment 6: heat diffusion PDE surrogate
│
├── plots/                                 # Auto-generated output plots (created at runtime)
└── README.md
```

---

## Requirements

### Python Version

Python 3.9 or higher is recommended.

### Jupyter Environment

All experiments are implemented as **Jupyter Notebooks** (`.ipynb`). You will need either JupyterLab or classic Jupyter Notebook installed:

```bash
pip install jupyterlab        # recommended
# or
pip install notebook
```

Launch the environment from the project root:

```bash
jupyter lab
# or
jupyter notebook
```

> **Important:** Always launch Jupyter from the **project root directory** so that local imports (`functions`, `samplers`, `interval_builder`, `envelope_builder`) resolve correctly. Do **not** open notebooks by double-clicking them from a file browser.

### Installing Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas scipy matplotlib seaborn tqdm scikit-learn torch plotly pyomo
```

### Full Dependency List

| Package | Min. Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26.4 | Numerical arrays and operations |
| `pandas` | ≥ 2.2.2 | Tabular results and CSV export |
| `scipy` | ≥ 1.13.1 | QMC samplers (Sobol, LHS), `norm` distribution |
| `matplotlib` | ≥ 3.9.0 | Inline plotting (`%pylab inline`) and figure export |
| `seaborn` | ≥ 0.13.2 | Statistical visualisations (Heat Diffusion notebook) |
| `plotly` | ≥ 5.22.0 | Interactive plots (Heat Diffusion notebook) |
| `tqdm` | ≥ 4.66.4 | Progress bars during iterative refinement |
| `tabulate` | ≥ 0.9.0 | Formatted console tables for result summaries |
| `scikit-learn` | ≥ 1.5.0 | GP regressor, MLP regressor, Random Forest, scalers, metrics |
| `scikit-optimize` | ≥ 0.10.2 | Gaussian EI acquisition function (`skopt.acquisition.gaussian_ei`) |
| `torch` | ≥ 2.3.1 | PyTorch MLP surrogate; uses MPS backend on Apple Silicon |
| `pyomo` | ≥ 6.7.3 | Symbolic modelling and McCormick relaxation via `pyomo.contrib.mcpp` |

> **Note on Pyomo McCormick plugin:** The `pyomo.contrib.mcpp.pyomo_mcpp.McCormick` class requires the `mc++` library to be compiled and linked with Pyomo. See the [Pyomo documentation](https://pyomo.readthedocs.io/en/stable/contributed_packages/mcpp.html) for installation instructions.

> **Note on inline plots:** All notebooks use the `%pylab inline` magic. Plots render directly inside the notebook cells. Interactive Plotly figures (Experiment 6) require a live Jupyter kernel with `plotly` installed; they will not render in static HTML exports.

---

## Running the Experiments

Open any notebook from the Jupyter interface and run all cells top-to-bottom using **Run → Run All Cells** (or `Shift+Enter` cell by cell). Each notebook is self-contained.

> Output plots are saved to `./plots/` subdirectories created automatically at runtime.

---

### Experiment 1 — Non-Hybrid Baseline Methods

**Notebook:** `Non-hybrid_Methods.ipynb`

Compares five sampling strategies (Sobol, LHS, Grid, Approach A, Approach B) on a suite of 1D benchmark functions using GP and MLP surrogates. Produces per-strategy RMSE/MAPE tables and sample distribution plots.

**Key configuration (edit in the configuration cell near the top):**

| Variable | Default | Description |
|---|---|---|
| `BUDGET` | `50` | Total number of training samples |
| `DOMAIN_SPLITS` | `10` | Number of envelope intervals for Approach A |
| `MIN_SAMPLES` | `2` | Minimum samples per interval |
| `FUNCTIONS` | (dict) | Functions to iterate over |

**Outputs:** Per-function RMSE/MAPE tables in cell outputs; sample and prediction plots saved to `./plots/`.

---

### Experiment 2 — Hybrid Methods: Choice of N (Domain Splits)

**Notebook:** `Hybrid_Methods-Choice_of_N.ipynb`

Runs the full hybrid pipeline (envelope warm start + iterative Bayesian refinement with EI/STD × gap weighting) across a sweep of domain split counts `N`. Reports heterogeneity metrics and RMSE for each `(function, N, strategy, surrogate)` combination.

**Key configuration:**

| Variable | Default | Description |
|---|---|---|
| `BUDGET` | `30` | Warm-start sampling budget |
| `REFINING` | `20` | Number of iterative refinement steps |
| `DOMAIN_SPLITS_LIST` | `[5, 10, 20]` | Values of N to sweep |
| `surrogate` | `'GP'` | Surrogate type: `'GP'` or `'NN'` |
| `xi` | `0.01` | EI exploration parameter |

**Outputs:** Envelope plots, per-iteration sampling frames in `./plots/it8/<func>_<experiment>/`, LaTeX-formatted RMSE summary tables in cell outputs.

---

### Experiment 3 — Hybrid Methods: Choice of M (Refinement Budget)

**Notebook:** `Hybrid_Methods_-_Choice_of_M.ipynb`

Mirrors Experiment 2 but sweeps over the **refinement budget** `M` (number of active learning steps) while holding `N` fixed.

**Key configuration:**

| Variable | Default | Description |
|---|---|---|
| `BUDGET` | `30` | Warm-start budget (fixed) |
| `REFINING_LIST` | `[10, 20, 40]` | Values of M (refinement steps) to sweep |
| `DOMAIN_SPLITS` | `10` | Fixed domain split count N |
| `surrogate` | `'GP'` | Surrogate type: `'GP'` or `'NN'` |

**Outputs:** Same structure as Experiment 2 — per-iteration plots, RMSE tables, LaTeX summaries in cell outputs.

---

### Experiment 4 — 2D Langermann Function

**Notebook:** `Langermann-2D_-_latest.ipynb`

Applies the envelope-guided sampling framework to the **2D Langermann function** defined over `[0, 10]²`. Extends the McCormick relaxation machinery to 2D domains and evaluates surrogate quality on a held-out grid.

**Key configuration:**

| Variable | Default | Description |
|---|---|---|
| `BUDGET` | `50` | Total 2D sample budget |
| `DOMAIN` | `(0, 10)` | Per-axis domain bounds |
| `GRID_SPLITS` | `5` | Splits per axis (total cells = GRID_SPLITS²) |

**Outputs:** 2D contour plots of true function, surrogate predictions, and gap heatmaps rendered inline.

---

### Experiment 5 — 2D Schwefel Function

**Notebook:** `Schwefel-2D.ipynb`

Same pipeline as Experiment 4 applied to the **2D Schwefel function** defined over `[-500, 500]²`.

**Key configuration:**

| Variable | Default | Description |
|---|---|---|
| `BUDGET` | `50` | Total 2D sample budget |
| `DOMAIN` | `(-500, 500)` | Per-axis domain bounds |
| `GRID_SPLITS` | `5` | Splits per axis |

**Outputs:** 2D contour/surface plots, gap heatmaps, RMSE comparison tables in cell outputs.

---

### Experiment 6 — Heat Diffusion PDE Surrogate

**Notebook:** `HeatDiffusion_-_OG.ipynb`

Builds a surrogate model for a **1D transient heat diffusion PDE** solved via the FTCS finite-difference scheme. The surrogate (MLP or GP) maps `(x, t)` inputs to temperature `T(x, t)`.

**Key configuration:**

| Variable | Default | Description |
|---|---|---|
| `cfg.L` | `0.1` | Rod length (m) |
| `cfg.D` | `4.25e-6` | Thermal diffusivity (m²/s) |
| `cfg.N` | `100` | Spatial grid points |
| `cfg.t_end` | `100.0` | Simulation end time (s) |
| `cfg.T_left / T_right` | `90.0` | Boundary temperatures (°C) |
| `cfg.T_init` | `20.0` | Initial temperature (°C) |
| `train_frac` | `0.65` | Fraction of timesteps used for training |

**Outputs:** Training loss curves, spatial temperature profile plots, RMSE/MAE/R² metrics on all data splits. Interactive Plotly figures render inline in the notebook (requires a live kernel).

---

## Output Directory

All figures are saved under `./plots/`. Subdirectory structure:

```
plots/
├── it8/                          # Hybrid method per-iteration frames
│   └── <func>_<experiment>/
│       ├── sampling/frame_*.png
│       └── ei/frame_*.png
├── output_analysis_nonhyb/       # Non-hybrid prediction plots
└── sample_analysis_nonhyb/       # Non-hybrid sample distribution plots
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{yourname2024mccormick,
  author = {Your Name},
  title  = {McCormick Envelope-Guided Adaptive Sampling for Surrogate Modelling},
  year   = {2024},
  url    = {https://github.com/your-username/your-repo-name}
}
```

---

## License

MIT License — see `LICENSE` for details.
