# %% [markdown]
# # Plot Pearsons correlation heatmap per cell for all markers

# %%
import numpy as np
import pandas as pd
import time
import os
import sys
from tqdm import tqdm, trange
from datetime import date
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
# from dask_ml.preprocessing import MinMaxScaler
import dask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nd2
from skimage import exposure, restoration
from joblib import Parallel, delayed
import torch
from cellpose import models, core, plot
import cv2
import scipy
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from skimage.morphology import closing, square
from skimage.measure import label
import scanpy as sc
import anndata
from anndata import AnnData
from pathlib import Path

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi = 300)

# sns.set(font_scale = 1.5)
sns.set_style('whitegrid')
np.random.seed(0)

# %% [markdown]
# # directories and inputs

# %%
# Get script directory to make paths relative to script location
SCRIPT_DIR = Path(__file__).parent

# drugs vs control
dfPaths = pd.read_excel(SCRIPT_DIR / ".." / "data" / "06 Plate 006 static compare" / "02_plate_drug_vs_ctrl_cycles1-5.xlsx")
print(dfPaths)

# folder to save graph screenshots
screenshotSavePath = SCRIPT_DIR / ".." / "figures"
assert screenshotSavePath.exists(), f"Path does not exist: {screenshotSavePath.absolute()}"

# identifier columns
idCols = ['Z', 'Y', 'X', 'CellLabel', 'CellRegion', 'Stim', 'FOV', 
          'MaskCytoLabel', 'MaskNucLabel', 'MaskNucMembLabel', 
          'NucCount', 'Cycle', 'NucCytoRatio', 'TimeStamp', 'Drug', 'AntiCorr']

# %% [markdown]
# # Delay execution if necessary

# %%
# time.sleep(1 * 60 * 60) # sec

# %% [markdown]
# # Compute median p65 for single cells in each FOV and concat

# %%
# reformat each FOV dataframe to single cell and concat
def reformatDfSingleCell(path, drug, timeStamp, idCols): 
    
    df = pd.read_pickle(path)
    # check if empty
    if df.size == 0:
        return None
    
    # # if no p65 RNA found
    # if 'p65 RNA Dots' not in df.columns:
    #     df['p65 RNA Dots'] = 0 # no signal
    # if 'GAPDH RNA Dots' not in df.columns:
    #     df['GAPDH RNA Dots'] = 0 # empty
        
    # compute single cell. per cell region
    markerNames = df.drop(columns = idCols, errors = 'ignore').columns
    # remove RNA intensities and IKBa
    markerNames = [m for m in markerNames if 'IKBa' not in m]
    markerNames = [m for m in markerNames if 'RNA Intensity' not in m]

    aggForm = {}
    aggForm['Stim'] = 'first'
    aggForm['FOV'] = 'first'
    for ii, marker in enumerate(markerNames):
        if 'RNA' in marker:
            aggForm[marker] = 'sum'
        else: # protein
            aggForm[marker] = 'median'
 
    dfCell = df.groupby(['CellLabel']).agg(aggForm)
    # reformat
    dfCell.reset_index(drop = False, inplace = True)

    # # reformat to make cell regions the cols
    # dfRegion = dfCell.pivot(index = ['Stim', 'CellLabel', 'FOV'], 
    #             columns = 'CellRegion')

    # # rename cols to include which cell region the marker is in
    # cols = []
    # for ii, colNames in enumerate(dfRegion.columns.values):
    #     # join names
    #     joint = ' '.join(colNames)
    #     cols.append(joint)

    # dfRegion.columns = cols # assign joint names

    # dfRegion.reset_index(drop = False, inplace = True)
    # dfRegion.fillna(0, inplace = True) # background    
    
    # # if no signal
    # for jj, cellRegion in enumerate(['Cytosol', 'Nuclear Membrane', 'Nucleus']):
    #     if 'p65 Protein ' + cellRegion not in dfRegion.columns:
    #         dfRegion['p65 Protein ' + cellRegion] = 0

    # # ratio of nuclear/cytosol p65 protein. Offset by 1 to avoid divide by zero
    # dfRegion['NucCytoRatio'] = np.true_divide(dfRegion['p65 Protein Nucleus'] + 1, 
    #                                        dfRegion['p65 Protein Cytosol'] + 1)

    # # reformat
    # dfRegion = dfRegion.melt(id_vars = ['Stim', 'CellLabel', 'FOV', 'NucCytoRatio'], 
    #            var_name = 'CellRegion', value_name = 'p65 Protein')
    # dfRegion['CellRegion'] = dfRegion['CellRegion'].str.replace('p65 Protein ', '')
    # dfRegion = dfRegion[['CellLabel', 'NucCytoRatio']]
    
    # # # combine RNA and protein
    # dfCell = dfCell.merge(dfRegion, how = 'outer').drop_duplicates()

    # add drug and timestamp info
    dfCell['Drug'] = drug
    if not isinstance(timeStamp, int):
        # get time from file name
        timeStamp = path.stem.split('_')[0].split('-')[1]

    dfCell['TimeStamp'] = int(timeStamp)

    return dfCell

dfAll = []
for row1 in dfPaths.itertuples(): # each sample type

    dfFiles = [f for f in Path(row1.PklPath).glob('*.pkl') if row1.Drug in f.stem or \
               str(row1.WellLabel) in f.stem]
    if len(dfFiles) == 0:
        continue

    # read in parallel
    dfSample = Parallel(n_jobs = 10, prefer = 'threads', verbose = 10)\
    (delayed(reformatDfSingleCell)(path = fileName, drug = row1.Drug, timeStamp = row1.Time,
                                   idCols = idCols)
    for _, fileName in enumerate(dfFiles))
        
    dfAll.extend(dfSample)

dfCell = pd.concat(dfAll)
dfCell

# %%
dfCell.groupby(['Drug']).agg({'CellLabel': 'nunique'}) # number of cells per drug

# %%
dfCell.describe()

# %% [markdown]
# # Plot Pearsons heatmap per cell among all markers. Separate plots per drug and timepoint

# %%
# compute Pearsons corr b/t RNA and protein per cell
def computePearsonsCorr(df, m1, m2):
    x = df[m1]
    y = df[m2]
    # stat, _ = scipy.stats.pearsonr(x = x, y = y)

    # compute pearsons correlation
    top = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    bot = np.sqrt(np.sum((x - np.mean(x))**2) * np.sum((y - np.mean(y))**2))
    pcc = np.true_divide(top, bot)
    
    return pcc

def saveFigure(fig, filename): # save figure with descriptive name
    fileOut = filename if filename.endswith('.png') else filename + '.png'
    fileOut = os.path.join(screenshotSavePath, fileOut)
    fig.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)
    print(f"Saved: {fileOut}")
    return None

# %%
# rename cols
for ii, colName in enumerate(dfCell.columns): # each col

    if 'Abs' in colName:
        dfCell.rename(columns = {colName: colName.replace('Abs', 'Proteins').strip()}, 
        inplace = True)

# %%
#  get list of all markers
markers = dfCell.drop(columns = idCols, errors = 'ignore').columns

for ii, drug in enumerate(dfCell['Drug'].unique()): # each drug

    dfDrug = dfCell.loc[dfCell['Drug'] == drug]

    for jj, timeStamp in enumerate(dfDrug['TimeStamp'].unique()):

        dfTime = dfDrug.loc[dfDrug['TimeStamp'] == timeStamp]

        # compute pearsons by cell for pairwise markers
        pcc = dfTime[markers].corr(method = 'pearson')

        # plot
        fig, ax = plt.subplots(dpi = 300)
        ax.set_title(drug + ' Time ' + str(timeStamp) + ' mins')
        if jj == 0: # first
            showCbar = True
        else:
            showCbar = False
        sns.heatmap(data = pcc, cmap = 'coolwarm', ax = ax, cbar = showCbar, 
                    cbar_kws={'label': 'Pearsons Correlation'})
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
        if jj > 0:
            ax.set_yticks([])
            ax.set_xticks([])

        # Create descriptive filename
        safe_drug = drug.replace(' ', '_').replace('/', '_')
        filename = f'Fig_S6_pearson_correlation_{safe_drug}_time_{timeStamp}mins'
        saveFigure(fig, filename)



