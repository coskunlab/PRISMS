# %% [markdown]
# # Plot OT2 temp control from data

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
dfTemp = pd.read_csv(SCRIPT_DIR / ".." / "data" / "Parhelia OT2" / "Patricia Pipetting" / "protocols" / "Temp Module" / "14May2025_temp_data.csv")
dfTemp.rename(columns={'TempSet': 'Set Temperature',
                       'TempMeasure': 'Measured Temperature'}, inplace=True)

screenshotSavePath = SCRIPT_DIR / ".." / "figures"
assert screenshotSavePath.exists(), f"Path does not exist: {screenshotSavePath.absolute()}"


# %% [markdown]
# # Plot temps

# %%
def saveFigure(fig, filename): # save figure with descriptive name
    fileOut = filename if filename.endswith('.png') else filename + '.png'
    fileOut = os.path.join(screenshotSavePath, fileOut)
    plt.gcf()
    plt.savefig(fileOut, dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)
    print(f"Saved: {fileOut}")

# %%
# reformat dataframe
dfSub = dfTemp.melt(id_vars=['TimeMins'], var_name='Type', value_name='Temp')

# %%
x = 'TimeMins'
y = 'Temp'
hue = 'Type'

fig, ax = plt.subplots(dpi = 300)
sns.lineplot(data = dfSub, x = x, y = y, ax = ax, hue = hue)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

ax.set_xlabel('Time (mins)')
ax.set_ylabel('Temperature (C)')
ax.set_title('Thermal Module Temperature Check')

saveFigure(fig, 'Fig_S1_thermal_module_temperature_check')


