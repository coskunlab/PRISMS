# %% [markdown]
# # Compute Pearsons single cell to compare across melanoma types

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
import dask
import matplotlib
import matplotlib.pyplot as plt
import nd2
from skimage import exposure, restoration
from joblib import Parallel, delayed
import torch
from cellpose import models, core, plot
import cv2
import scipy
import seaborn as sns
from datetime import date, datetime
import matplotlib.pylab as pl
from sklearn.preprocessing import MinMaxScaler
import math
from pathlib import Path

sns.set(font_scale=2)

# %% [markdown]
# # directories and inputs

# %%
# Get script directory to make paths relative to script location
SCRIPT_DIR = Path(__file__).parent

# data file path
dataPath = SCRIPT_DIR / ".." / "data" / "Sample 11L" / "15May2024 slide 021 section 1 cycle 5" / "08 PKL single cell"
assert dataPath.exists(), f"Path does not exist: {dataPath.absolute()}"

# folder to save screenshots
screenshotSavePath = SCRIPT_DIR / ".." / "figures"
assert screenshotSavePath.exists(), f"Path does not exist: {screenshotSavePath.absolute()}"

# cell identifiers for dataframe
idCols = ['Z', 'Y', 'X', 'CellLabel', 'FOV', 'Cycle', 'CellRegion', 'MaskCytoLabel', 'MaskNucLabel']

# %% [markdown]
# # Delay execution (optional)

# %%
# time.sleep(1/2 * 60 * 60) # sec

# %% [markdown]
# # Read all dataframes and compute Pearsons for single cell

# %%
# compute Pearsons for single marker pair
def computePearsonsPair(df, m1, m2):
    
    gray1 = df[m1]
    mean1 = np.nanmean(gray1)
    gray2 = df[m2]
    mean2 = np.nanmean(gray2)
    
    top = np.multiply(gray1 - mean1, 
                     gray2 - mean2)
    top = np.nansum(top)
    
    bot = np.nansum(np.square(gray1 - mean1))
    bot *= np.nansum(np.square(gray2 - mean2))
    bot = np.sqrt(bot)
    
    corr = np.true_divide(top, bot)
    
    return corr

# compute Pearsons for all marker pairs
def computePearsonsPerCell(df, markerPairs):
    
    dfPear = {}
    for ii, pair in enumerate(markerPairs): # each marker pair

        m1, m2 = pair.split('_') 
        
        pcc = computePearsonsPair(df = df, m1 = m1, m2 = m2) # per FOV across cells
        dfPear[pair] = pcc
    
    dfPear = pd.DataFrame.from_records(dfPear, index = [0])

    # # nans are 0 correlation
    # dfPear.fillna(0, inplace = True)

    return dfPear

# reformat each FOV dataframe to single cell and concat
def reformatDfSingleCell(path, idCols): 
    
    df = pd.read_pickle(path)
    # check if empty
    if df.size == 0:
        return None
    
    # # test first few cells for now
    # df = df.head(10000)
    # # print(df['CellLabel'].nunique(), 'Cells')

    # only choose first 3 cycles
    df.drop(columns = ['CD34', 'Calponin', 'VCAM1'], inplace = True)
    
    # read parquet file directly w/o dask
    markers = df.drop(columns = idCols, errors = 'ignore').columns.tolist() # markers
    
    # create list of unique marker pairs
    markerPairs = [m1 + '_' + m2 for ii, m1 in enumerate(markers) 
                  for jj, m2 in enumerate(markers) 
                  if jj > ii]
        
    # compute Pearsons per cell for all marker pairs
    dfCell = df[['FOV', 'CellLabel', *markers]]
    
    # OR compute sum signal per cell. Then compute Pearsons over cells instead of pixels w/in cells
    # dfCell = dfCell.groupby(['FOV', 'CellLabel']).sum() # sum protein signal per cell
    # compute pearsons per cell for all cells
    dfPear = dfCell.groupby(['FOV', 'CellLabel']).apply(computePearsonsPerCell, 
                                                        markerPairs = markerPairs)
    dfPear.reset_index(drop = False, inplace = True)

    # drop random cols
    dfPear.drop(columns = ['level_2'], inplace = True, errors = 'ignore')

    # # compute pearsons per cell for all pairs of markers
    # dfPear = dfCell.groupby(['FOV', 'CellLabel']).apply(computePearsonsPerCell, 
    #                                                     markerPairs = markerPairs) # per cell across markers
    
    # dfPear = []
    # for ii, pair in enumerate(markerPairs):
        
    #     m1, m2 = pair.split('_')
    #     # Pearsons = nan if one marker is all zeros
    #     # pcc = dfCell.groupby(['FOV', 'CellLabel']).apply(computePearsonsPair, m1 = m1, m2 = m2)        
    #     # pcc.rename(pair, inplace = True) # rename Series
    #     # dfPear.append(pcc) # each single marker pair      
        
    #     pcc = computePearsonsPair(df = dfCell, m1 = m1, m2 = m2) # per FOV across cells
    #     dfPear.append(pd.Series(pcc, name = pair))
        
    # dfPear = pd.concat(dfPear, axis = 1) # concat cols. all pairwise markers
    
    # # reformat
    # dfPear.reset_index(drop = True, inplace = True)

    return dfPear

dfFiles = [f for f in dataPath.glob('*.pkl')]
    
# read in parallel
dfAll = Parallel(n_jobs = 1, prefer = 'threads', verbose = 10)\
(delayed(reformatDfSingleCell)(path = fileName, 
                                idCols = idCols) \
    for _, fileName in enumerate(dfFiles))

# concat for all samples
dfCell = pd.concat(dfAll)
dfCell

# %% [markdown]
# # Plot boxplot of Pearsons and compute stats

# %%
def saveFigure(fig, filename): # save figure with descriptive name
    fileOut = filename if filename.endswith('.png') else filename + '.png'
    fileOut = os.path.join(screenshotSavePath, fileOut)
    fig.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)
    print(f"Saved: {fileOut}")
    return None

# %%
# reformat 
dfSub = dfCell.drop(columns = ['FOV', 'CellLabel'], errors = 'ignore')
dfSub = dfSub.melt(var_name = 'MarkerPair', value_name = 'Pearsons')

# %%
# plot all pearsons values
x = 'MarkerPair'
y = 'Pearsons'

fig, ax = plt.subplots(dpi = 300, figsize = (20, 10))
sns.boxplot(x = x, y = y, data = dfSub, ax = ax)
ax.set_ylabel('Pearsons Correlation (AU)', fontsize = 48)
ax.set_xlabel('Marker Pair', fontsize = 48)
ax.set_title('Pairwise Colocalization', fontsize = 48)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

saveFigure(fig, 'Fig_6a_pairwise_colocalization_boxplot')

# %%
# # plot each marker against all other pairs
# for ii, m1 in enumerate(tqdm(markers)): # marker 1

#     # plot
#     x = 'MarkerPair'
#     y = 'Pearsons'
#     hue = 'Stage'
#     hueOrder = ['Large Nevus', 'Small Melanoma', 'Large Melanoma']
    
#     # splice dataframe
#     dfSub = dfCell.loc[(dfCell['Marker1'].str.contains(m1)) |
#               (dfCell['Marker2'].str.contains(m1))]
    
#     fig, ax = plt.subplots(dpi = 300, facecolor = 'white')
#     try:
#         g = sns.boxplot(data = dfSub, x = x, y = y, ax = ax, hue = hue, hue_order = hueOrder)
        
#     except:
#         continue
        
#     g = g.axes

#     # add stats
#     boxPairs = []
#     for jj, pair in enumerate(g.get_xticklabels()): # each marker pair
        
#         markerPair = pair.get_text()
        
#         for kk, g1 in enumerate(dfSub[hue].unique()): # each hue group
            
#             for ll, g2 in enumerate(dfSub[hue].unique()): # other hue group
            
#                 if kk <= ll: # same marker
#                     continue
                    
#                 # check if group is empty
#                 single = dfSub.loc[dfSub['MarkerPair'].str.contains(markerPair)]
#                 single.dropna(axis = 0, how = 'any', subset = ['Pearsons'], inplace = True)
#                 # if single[hue].unique().shape != dfSub[hue].unique().shape: # at least 1 empty group
#                 #     continue
#                 if single[hue].str.contains(g1).any() and single[hue].str.contains(g2).any(): # valid group
#                     boxPairs.append(((markerPair, g1), (markerPair, g2)))
                
#     if len(boxPairs) > 0:
#         add_stat_annotation(g, data = dfSub, x = x, y = y, hue = hue, hue_order = hueOrder, 
#                            box_pairs = boxPairs, test = 'Mann-Whitney', loc = 'inside', 
#                            comparisons_correction = 'bonferroni')
    
#     # rename underscore in y labels
#     labels = []
#     for jj, pair in enumerate(g.get_xticklabels()):

#         markerPair = pair.get_text()
#         # make m1 the first name
#         p1, p2 = markerPair.split('_')
#         if p2 == m1: # reverse order
#             markerPair = p2 + '_' + p1
        
#         markerPair = markerPair.replace('_', ' & ')
#         labels.append(markerPair)

#     # set new labels
#     g.set_xticklabels(labels, rotation = 45, fontsize = 8)

#     # ax.set_ylim(-1, 1) # Pearsons limits
#     ax.set_xlabel('Pairwise Markers')
#     ax.set_title(m1 + ' Pearsons Single Cell')
    
#     # move legend outside
#     ax.legend(bbox_to_anchor = [1, 1], loc = 'upper left')
    
#     # save fig
#     now = datetime.now() # current date and time
#     date_time = now.strftime("%d%b%Y_%H%M%S")
#     fileOut = date_time + '.png'
#     fileOut = os.path.join(screenshotSavePath, fileOut)
#     plt.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)


