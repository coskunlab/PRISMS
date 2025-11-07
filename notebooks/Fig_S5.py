# %% [markdown]
# # Plot single cell RNA expression. Compare tissues

# %%
import numpy as np
import pandas as pd
import time
import os
import sys
from tqdm import tqdm, trange
from datetime import date, datetime
import nd2reader
from joblib import Parallel, delayed
import tifffile as tf
import xml.etree.ElementTree as ET
import napari
import skimage
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.filters import try_all_threshold
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask
import matplotlib
import matplotlib.pyplot as plt
import nd2
from skimage import exposure, restoration
from joblib import Parallel, delayed
import torch
import cellpose
import cv2
import scipy
from skimage.morphology import closing, square
from skimage.measure import label
import seaborn as sns
import math
import sklearn
from sklearn.neighbors import KDTree
import networkx
from networkx.algorithms.components.connected import connected_components
from pathlib import Path
import h5py

sns.set(font_scale=2)
sns.set_style("whitegrid")

# %% [markdown]
# # user directories and inputs

# %%
# Get script directory to make paths relative to script location
SCRIPT_DIR = Path(__file__).parent
pklPath = SCRIPT_DIR / ".." / "data" / "Human tonsil HuFPT161 SW5" / "21Feb2025 slide 034 cycle 2 IF Squid" / "09 PKL single cell"
assert pklPath.exists(), f"Path does not exist: {pklPath.absolute()}"

screenshotSavePath = SCRIPT_DIR / ".." / "figures"
assert screenshotSavePath.exists(), f"Path does not exist: {screenshotSavePath.absolute()}"

idCols = ['Z', 'Y', 'X', 'CellLabel', 'MaskNucLabel', 'MaskCytoLabel', 'TumorStage', 'TumorType', 'CellRegion', 'Cycle', 'FOV']

# %% [markdown]
# # Delay execution (optional)

# %%
# time.sleep(20 * 60) # sec

# %% [markdown]
# # Reformat single cell dataframes

# %%
def reformatDfSingleCell(fileName, idCols):

    df = pd.read_pickle(fileName)
    cols = df.drop(columns = idCols, errors = 'ignore').columns.tolist()
    # cols = df.columns.tolist()

    aggForm = {}
    for col in cols:
        aggForm[col] = 'sum'

    dfCell = df.groupby(['CellLabel']).agg(aggForm) # sum per cell
    dfCell.reset_index(inplace = True, drop = False)

    return dfCell

pklFiles = [f for f in Path(pklPath).glob("*.pkl")]
print(f"Found {len(pklFiles)} pkl files")

dfCell = Parallel(n_jobs=-1, prefer = 'threads', verbose = 10)\
    (delayed(reformatDfSingleCell)(fileName = fileName, idCols = idCols) for fileName in pklFiles)

dfCell = pd.concat(dfCell)
dfCell.reset_index(inplace = True, drop=True)
dfCell

# %% [markdown]
# # Plot compare single cell expression

# %%
def saveFigure(fig, filename): # save figure with descriptive name
    fileOut = filename if filename.endswith('.png') else filename + '.png'
    fileOut = os.path.join(screenshotSavePath, fileOut)
    plt.gcf()
    plt.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)
    print(f"Saved: {fileOut}")

# %%
# reformat df for plotting
dfSub = dfCell.drop(columns = ['CellLabel', 'Hoescht Protein']).copy(deep = True) # drop Hoechst for RNA counts
# dfSub.reset_index(drop = False, inplace=True)
dfSub = dfSub.melt(var_name='Marker', value_name='Expression')
dfSub['Marker'] = dfSub['Marker'].str.replace(' Protein', '')
dfSub['Marker'] = dfSub['Marker'].str.replace(' RNA Dots', '')
dfSub

# %%
x = 'Marker'
y = 'Expression'
hue = 'Marker'

fig, ax = plt.subplots(dpi = 300)
sns.boxplot(data = dfSub, x = x, y = y, hue = hue, ax = ax)
ax.set_ylabel('Expression Per Cell (AU)')
# ax.set_ylabel('RNA Counts Per Cell')
# ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

saveFigure(fig, 'Fig_S5_protein_expression_per_cell_boxplot')

# %%
# plot cell wise pariwise pearson's correlation
def computePearsonCorr(arr1, arr2):
    
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    
    numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
    denominator = np.sqrt(np.sum((arr1 - mean1) ** 2) * np.sum((arr2 - mean2) ** 2))
    
    if denominator == 0:
        return 0  # or np.nan if either image is constant
    
    return numerator / denominator

markers = dfCell.drop(columns=['CellLabel']).columns.tolist()
dfCorr = pd.DataFrame(index = markers, columns = markers, dtype = float)

for ii, m1 in enumerate(markers): # each marker

    for jj, m2 in enumerate(markers): # each marker

        if jj >= ii: # pairwise w/o repeats

            corr = computePearsonCorr(dfCell[m1], dfCell[m2])
            dfCorr.iloc[ii, jj] = corr
            dfCorr.iloc[jj, ii] = corr
# plot
markers = [m.replace(' Protein', '') for m in markers]
fig, ax = plt.subplots(dpi = 300)
sns.heatmap(dfCorr, ax=ax, cmap = 'coolwarm', 
            cbar_kws={'label': 'Pearsons Correlation'})
ax.set_xticklabels(markers, rotation = 45)
ax.set_yticklabels(markers, rotation = 0)
ax.set_title('Pairwise Pearson Correlation')
saveFigure(fig, 'Fig_S5_pearson_correlation_heatmap')


