# GSR-PPI

This repository contains code for the manuscript "Graph-based spatial proximity of super-resolved protein-protein interactions predict cancer drug responses in single cells". Codes are run under the specified Anaconda environment.

To set up environments, run the following command: `conda env create -f environment.yml`

Example data can be found at: https://figshare.com/projects/Signaling_Project_PLA/195958 


# Citation

Please cite: Zhang, N. et al. Graph-Based Spatial Proximity of Super-Resolved Protein–Protein Interactions Predicts Cancer Drug Responses in Single Cells. Cel. Mol. Bioeng. https://doi.org/10.1007/s12195-024-00822-1 (2024) doi:10.1007/s12195-024-00822-1.

## Repository Overview

This repository contains analysis scripts for spatial transcriptomics and single-cell molecular imaging data. The primary purpose is to process single-cell data from microscopy experiments, perform statistical analysis, and generate publication-quality figures.

### Key Technologies

*   **Data Analysis:** `pandas`, `numpy`, `scikit-learn`, `scipy`
*   **Visualization:** `matplotlib`, `seaborn`, `napari`
*   **Image Processing:** `scikit-image`, `opencv-python`, `cellpose`
*   **Microscopy Data:** `nd2reader`, `nd2`, `tifffile`
*   **Spatial Analysis:** Graph Neural Networks (`torch-geometric`), `networkx`, `spatial-statistics`
*   **Parallel Processing:** `joblib`, `dask`
*   **Single-Cell Analysis:** `scanpy`, `anndata`
*   **Custom Packages:**
    *   `pixelator`: A package from Pixelgen Technologies for pixel-level spatial omics.
    *   `py_sphere`: A package for spherical geometry utilities.

### Directory Structure

```
.
├── data/                          # Experimental datasets (PKL files, Excel metadata)
├── figures/                       # Generated PNG outputs from figure scripts
├── notebooks/                     # Analysis scripts and package modules
│   ├── Fig_4_confocal.py         # Confocal microscopy RNA expression analysis
│   ├── Fig_4_widefield.py        # Widefield microscopy RNA expression analysis
│   ├── Fig_5.py                  # RNA expression analysis
│   ├── Fig_6a.py                 # Pairwise colocalization analysis
│   ├── Fig_S1.py                 # Thermal module temperature validation
│   ├── Fig_S4.py                 # Autofocus gradient analysis
│   ├── Fig_S5.py                 # Protein expression analysis
│   ├── Fig_S6.py                 # Per-cell Pearson correlation heatmap (computationally intensive)
│   ├── GNN/                      # Graph Neural Network module for spatial transcriptomics
│   ├── pixelator/                # Pixelgen Technologies spatial omics package
│   └── py_sphere/                # Spherical geometry utilities
└── environment.yml                # Conda environment definition
```

## Building and Running

### Environment Setup

To set up the Anaconda environment, run the following command:

```bash
conda env create -f environment.yml
```

### Running Figure Generation Scripts

All figure generation scripts are located in the `notebooks/` directory and output to `figures/`. To run a specific script:

```bash
python notebooks/Fig_4_confocal.py
```

To regenerate all figures:

```bash
for script in notebooks/Fig_*.py; do python "$script"; done
```

**Performance Notes**:

*   Most scripts complete in 30 seconds to 3 minutes.
*   `Fig_S6.py` is computationally intensive and can take 30+ minutes.
*   Scripts use `joblib` with a threading backend for parallelization.

## Development Conventions

### Code Style

*   The codebase is not strictly formatted, but new code should follow standard Python conventions (PEP 8).
*   Use meaningful variable names and add comments to explain complex logic.

### Data Processing Pipeline

The typical workflow for analysis scripts is as follows:

1.  Load `.pkl` files (pickled DataFrames with single-molecule spatial data).
2.  Aggregate data per-cell using `groupby` operations.
3.  Apply statistical analysis (correlations, colocalization, etc.).
4.  Generate visualizations (boxplots, heatmaps, scatter plots).
5.  Save figures as PNG to the `figures/` directory.

### Path Resolution

All scripts use relative paths from the script location:

```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
pklPath = SCRIPT_DIR / ".." / "data" / "experiment_name" / "subfolder"
screenshotSavePath = SCRIPT_DIR / ".." / "figures"
```

### Testing

There is no automated test suite in this repository. To verify changes:

1.  Run all figure generation scripts.
2.  Check that PNG outputs are created in the `figures/` directory.
3.  Visually inspect the generated figures for expected patterns.
