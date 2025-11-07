# %% [markdown]
# # Plot autofocus from DAPI

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
import dask_image.imread
from datetime import datetime

sns.set(font_scale=2)
sns.set_style("whitegrid")

# %% [markdown]
# # Test if napari is working 

# %%
import napari
from skimage import data

try:
    viewer = napari.view_image(data.astronaut(), rgb=True)

except:
    print('Napari not working. Re install package and restart kernel')
    
else:
    viewer.close()

# %% [markdown]
# # user directories and inputs

# %%
# Get script directory to make paths relative to script location
SCRIPT_DIR = Path(__file__).parent
filePath = SCRIPT_DIR / ".." / "data" / "EA.hy926 Cells Navinci Pairs" / "5.1.25 EA.hy926 B2-B3 Round 2" / "00 RAW" / "B3-2_y001_DAPI_zScan.nd2"
assert filePath.exists(), f"Path does not exist: {filePath.absolute()}"

screenshotSavePath = SCRIPT_DIR / ".." / "figures"
assert screenshotSavePath.exists(), f"Path does not exist: {screenshotSavePath.absolute()}"


# %% [markdown]
# # Compute autofocus and plot curve

# %%
def saveFigure(fig, filename): # save figure with descriptive name
    fileOut = filename if filename.endswith('.png') else filename + '.png'
    fileOut = os.path.join(screenshotSavePath, fileOut)
    plt.gcf()
    plt.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)
    print(f"Saved: {fileOut}")

# %%

# compute laplacian for each z plane in parallel
def computeLapVar(plane):
    
    var = cv2.Laplacian(plane, cv2.CV_64F, ksize = 31)
    var = np.var(var)
    
    return var

# find focus plane via Laplacian variance
def findFocusLapVar(subStack):
    
    # lapVar = Parallel(n_jobs = -1, prefer = 'threads', verbose = 0)\
    # (delayed(computeLapVar)(subStack[ii, :, :].compute()) for ii in range(subStack.shape[0]))

    # use dask instead
    lazy_lap_var = dask.delayed(computeLapVar)
    # lapVar = da.map_blocks(computeLapVar, imgStack, dtype = float, chunks = (imgStack.chunksize[0], 0, 0))
    lapVar = []
    for jj in range(subStack.shape[0]): # each z plane
        plane = subStack[jj, :, :] # YX
        var = lazy_lap_var(plane)
        var = da.from_delayed(var, shape = (1,), dtype = float)
        lapVar.append(var) # each scalar

    lapVar = da.concatenate(lapVar).compute() # single axis Z length
    
    idxFocus = np.argmax(lapVar)
    xRange = np.arange(0, len(lapVar))
    
    # compute steepest gradient in variance to find focus plane
    grad = np.gradient(lapVar)
    grad = np.square(grad)
    
    # extract peaks of gradient
    mean = np.mean(grad)
    # peaks with min horizontal distance
    peaks, props = scipy.signal.find_peaks(x = grad, height = mean, distance = len(lapVar) // 3)
    heights = props['peak_heights']
    # tallest = np.argsort(-1 * heights)
    # peaks = [peaks[ii] for ii in tallest]
    # print('Peaks', peaks)
    
    # plot gradient and peaks
    fig, ax = plt.subplots(dpi=300, figsize=(5, 25))  # Adjust figure size for narrow and tall plot
    ax.plot(grad, xRange, label='Gradient')  # Transpose the plot
    ax.axvline(mean, label='Gradient Mean', color='black', linestyle='--')
    ax.scatter(grad[peaks], xRange[peaks], label='Peaks', color='red')
    ax.set_xscale('log')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylabel('Z Plane')
    ax.set_xlabel('Gradient')
    
    # idxFocus = np.argmax(grad) + 2
    if len(peaks) == 0: # no peaks found
        print('No peaks found...')
        idxFocus = len(lapVar) // 2 # middle
        
    else:
        idxFocus = peaks[0] + 2 # tallest peak    
        
    if idxFocus > len(lapVar) - 2: # exceeds length, out of bounds
        idxFocus = len(lapVar) - 2
        
    return idxFocus, peaks, fig

# %%
imgStack = nd2.imread(filePath, dask = True)

# compute focus plane via blur detection
focusStep = 2 # step size, um
focusCount = 150  # number of z planes
dimZ = 40

# zRange = focusStep * focusCount # total range, um
# zRange = np.linspace(z - zRange / 2, z + zRange / 2, dimZ)    
idxFocus, peaks, fig = findFocusLapVar(imgStack) # all peaks
# select closest peaks to prev
if not isinstance(peaks, int) and len(peaks) > 1: # multiple peaks
    print('Multiple peaks found. Finding nearest peak to middle...')
    diff = peaks - focusCount // 2
    diff = np.abs(diff)
    closest = np.argmin(diff)
    idxFocus = peaks[closest]

print('Focus plane is', idxFocus, '/', focusCount)

saveFigure(fig, 'Fig_S4_autofocus_gradient_analysis')

# %%
viewer = napari.Viewer()
viewer.add_image(imgStack, scale = [2, 0.108, 0.108], name = 'DAPI', colormap = 'blue', blending = 'additive')
viewer.scale_bar.visible = True
viewer.scale_bar.font_size = 36


