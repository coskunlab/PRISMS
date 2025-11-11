# %% [markdown]
# # Estimate current remaining time for current Squid acquisition
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime
import os
import sys
import tifffile as tf
from pathlib import Path
import json
import time
from IPython.display import clear_output
import urllib.parse
import shutil
import skimage
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import scipy
import pip
import cv2
import platform
import smtplib
from email.message import EmailMessage
import asyncio
import psutil
import dask_image.imread
from collections import defaultdict
from mpi4py import MPI

try:
    __import__('pytimedinput')
    __import__('basicpy')
except ImportError:
    pip.main(['install', 'pytimedinput'])
    pip.main(['install', 'basicpy'])

from pytimedinput import timedInput
from basicpy import BaSiC
from basicpy import datasets as bdata

'''
Running single sample manually:
command = f'python3 ./control/process_images_real_time.py {urllib.parse.quote(current_path)} \
            {self.overlapFrac} {self.NZ} {self.dimY} {self.dimX} {self.numFrames} {self.scan_coordinates_mm.shape[0]} \
                 {self.stitchMode} {urllib.parse.quote(str(self.localPath))}'
os.system(f"gnome-terminal -e 'bash -c \"{command};bash\"'")

command = f'python3 ./control/process_images_real_time.py /home/cephla/Desktop/Nicky/48%20NFkB%20gradient%20on%20chip/Data/01-3T3%20P26%20coverslip%20112B/23Jan2025%20cycle%204%20HCR/_2025-01-07_14-39-41.123247/0 0.15 40 3 3 1 26 MIP /home/cephla/Downloads/01-3T3%20P26%20coverslip%20112B/23Jan2025%20cycle%204%20HCR/_2025-01-07_14-39-41.123247/0'

python3 process_images_real_time.py /home/cephla/Desktop/coskun-lab/Nicky/52%20Breast%20Tissue%20Yesim%20Sunil/Data/577-2805_1%20primary%20tumor%2034%20months/28Mar2025%20slide%20005%20cycle%203%20IF/_2025-04-03_13-23-25.913464/0 0.15 40 53 66 1 1 BestFocus /home/cephla/Downloads/577-2805_1%20primary%20tumor%2034%20months/28Mar2025%20slide%20005%20cycle%203%20IF/_2025-04-03_13-23-25.913464/0

'''

#%%
# --- MPI Initialization ---
COMM = MPI.COMM_WORLD # Default communicator
RANK = COMM.Get_rank() # Get the rank (ID) of the current process
SIZE = COMM.Get_size() # Get the total number of processes
# print('Rank', rank, 'of size', size, 'started.')
# assert SIZE > 1, 'This script must be run with MPI (multiple processes).'
if SIZE == 1:
    print('WARNING: This script is not running with multiple nodes...')

# %% [markdown]
# # user directories and inputs
# basePath = Path(r'/storage/home/hcoda1/5/nzhang326/scratch/49 Cystic Fibrosis - Rabin/Data/24 well plate 003B/8Apr2025 Row A cycle 3 IF/')
script_dir = os.path.dirname(os.path.abspath(__file__))
basePath = Path(script_dir).parent
assert basePath.exists()

# only write macro if tiles are computed already
onlyWriteMacro = False

# raw TIFF folder
# remove %20 from path
# rawPath = sys.argv[1]
# rawPath = urllib.parse.unquote(rawPath)
# rawPath = r"Y:\coskun-lab\Nicky\49 Cystic Fibrosis - Rabin\Data\24 well plate 003A\13Feb2025 cycle 1 HCR and IF\_2025-02-13_12-58-00.275373\0"
# rawPath = Path(rawPath)
rawPath = [f for f in basePath.glob('*') if f.is_dir() and f.stem.startswith('_')]
assert len(rawPath) == 1
rawPath = rawPath[0] / '0'
assert rawPath.exists()

overlapFrac = 0.15
# print('Overlap fraction:', overlapFrac)

jsonFile = [f for f in rawPath.parent.glob('*.json')]
assert len(jsonFile) == 1
with open(jsonFile[0]) as f:
    meta = json.load(f)
    dimZ = meta['Nz']
    tileY = meta['Ny']
    tileX = meta['Nx']
nRepFrames = 1

# multipt coords
# numCoords = int(sys.argv[7])

# stitchMode = sys.argv[8]
# stitchMode = input('Stitch MIP or BestFocus? ')
stitchMode = 'MIP'

# local save path
# localPath = sys.argv[9]
# localPath = urllib.parse.unquote(localPath)
localPath = r''
localPath = Path(localPath)
assert localPath.exists()

# folder to save split TIFs for later stitching
rawTilePath = Path(basePath, f'01 TIF tiles raw {stitchMode}')
rawTilePath.mkdir(exist_ok = True)

# folder to save processed TIFs for later stitching
processedTilePath = Path(basePath, f'02 TIF tiles processed {stitchMode}')
processedTilePath.mkdir(exist_ok = True)

# folder to save stitched large images
largeImagePath = Path(basePath, '03 TIF stitched')
largeImagePath.mkdir(exist_ok = True)

# folder to render DAPI to check focus
renderPath = basePath / '04 PNG renders'
renderPath.mkdir(exist_ok = True)

# folder to export combined H5 file to
rawTifPath = basePath / '05 TIF raw'
rawTifPath.mkdir(exist_ok=True)

# read coords
coords = [f for f in basePath.glob('*.csv')]
if len(coords) > 1:
    coords.sort() # read latest
coords = pd.read_csv(coords[-1])
if len(coords['ID'].unique()) == 1: # boundary
    numCoords = 1
else:
    assert len(coords['ID'].unique()) == len(coords), 'Duplicate coords found'
    numCoords = len(coords)

# see if coords match to use coord names or not
if len(coords) == numCoords: # multipt coords
    useCoordNames = True
    # tiles to stitch 3D volumes
    volumeTilePath = basePath / '06 TIF tiles for 3D stitching Grid Collection'
    if RANK == 0: volumeTilePath.mkdir(exist_ok = True)

    # folder for 3D stitching volumes
    volumeStitchPath = basePath / '07 stitched 3D volumes Grid Collection'
    if volumeStitchPath.exists() and RANK == 0:
        # prompt user to delete folder
        # ans, timedOut = timedInput('Delete exis/ting 3D stitched volumes folder? (y/n) ', timeout = 60)
        # if not timedOut and ans == 'y':
        shutil.rmtree(volumeStitchPath) # delete current folder
    if RANK == 0: volumeStitchPath.mkdir(exist_ok=True)

else: # large scan
    useCoordNames = False

assert useCoordNames, 'This script is for multipoint tile scans only'

# %% [markdown]
# Copy files from local to network while waiting
def copyLocalFilesToNetwork(localPath, networkPath, ii, jj, coord):
    # print('Moving files from local to network...')
    localFiles = [f for f in Path(localPath).glob('*.tiff') if f.stem.startswith(f'{str(coord)}_{ii}_{jj}_')]
    if len(localFiles) == 0:
        return None

    localFiles.extend([f for f in Path(localPath).glob('*.txt')])
    for fileName in localFiles:
        # attempt copy and delete file
        dest = Path(networkPath) / fileName.name
        try:
            # shutil.copy2(fileName, dest)
            # os.remove(fileName)
            shutil.move(fileName, dest)
        except:
            print('Failed to move', fileName.name)
    # print('Finished moving files from local to network...')

# # get channels acquired
def getChannelsAcquired(localPath, rawPath):

    # rawFiles = [f for f in rawPath.glob('*.tiff')]
    rawFiles = getRawFiles(localPath, rawPath)
    channelFilters = []
    for ii, fileName in enumerate(tqdm(rawFiles)): # each raw file

        chan = fileName.stem.split('_')[4:]
        # for SRRF drop frame number
        if 'Frame' in chan:
            chan = chan[:-1] # drop last

        chan = '_'.join(chan) # complete channel name
        channelFilters.append(chan) # each selected channel

    channelFilters = np.unique(channelFilters) # unique channels
    channelFilters = np.sort(channelFilters)
    # assert len(channelFilters) > 0, 'No channels found in raw files'

    # if using RGB, use full white light first
    rgbMode = False
    if 'BF_LED_matrix_full_G' in channelFilters:
        channelFilters = [f for f in channelFilters if 'BF_LED_matrix_full_G' not in f]
        channelFilters = ['BF_LED_matrix_full_G'] + channelFilters
        rgbMode = True
    if 'BF_LED_matrix_full' in channelFilters:
        channelFilters = [f for f in channelFilters if 'BF_LED_matrix_full' not in f]
        channelFilters = ['BF_LED_matrix_full'] + channelFilters

    return channelFilters, rgbMode

# find all raw files so far
def getRawFiles(localPath, rawPath):
    rawFiles = [f for f in rawPath.glob('*.tiff')] # network path
    rawFiles.extend([f for f in localPath.glob('*.tiff')]) # local path
    return rawFiles

#%% stitching
# compute MIPs and export in parallel
# compute best focus planes
# compute laplacian for each z plane in parallel
def computeLapVar(plane):
    
    var = cv2.Laplacian(plane, cv2.CV_64F, ksize = 31)
    var = np.var(var)
    
    return var

# find focus plane via Laplacian variance
def findFocusLapVar(subStack):

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

    # xRange = np.arange(0, len(lapVar))
    
    # compute steepest gradient in variance to find focus plane
    grad = np.gradient(lapVar)
    grad = np.square(grad)
    
    # extract peaks of gradient
    thresh = np.percentile(grad, 50)
    # peaks with min horizontal distance
    peaks, props = scipy.signal.find_peaks(x = grad, height = thresh, distance = 1)
    heights = props['peak_heights']
    # tallest = np.argsort(-1 * heights)
    # peaks = [peaks[ii] for ii in tallest]
    # print('Peaks', peaks)
    
    # # plot gradient and peaks
    # fig, ax = plt.subplots(dpi = 300)
    # ax.scatter(xRange, lapVar, label = 'Variance')
    # ax.scatter(xRange, grad, label = 'Gradient')
    # ax.axhline(thresh, label = 'Peak Threshold')
    # ax.scatter(xRange[peaks], grad[peaks], label = 'Peaks')
    # ax.legend()
    
    # find peaks that are more consecutive. 
    # Score peaks based on if adjacent to other peaks
    # peak with consecutive neighbors is focus plane
    
    # idxFocus = np.argmax(grad) + 2
    if len(peaks) == 0:
        idxFocus = len(lapVar) // 2 # middle
        
    else:
        idxFocus = peaks[0] # tallest peak
        
    if idxFocus > len(lapVar) - 2: # exceeds length, out of bounds
        idxFocus = len(lapVar) - 2
        
    return idxFocus, peaks

# get camera type and orientation
def getCameraTypeOrientation(rawPath):

    # get first file
    rawFiles = getRawFiles(localPath, rawPath)
    fileName = rawFiles[0]
    # read first file
    plane = dask_image.imread.imread(fileName).squeeze()
    dimY, dimX = plane.shape
    if dimY == 2304:
        print('Hamamatsu camera detected')
        cameraType = 'Hamamatsu'
    elif dimY == 2084:
        print('Sony camera detected')
        cameraType = 'Sony'
    else:
        sys.exit('Unknown camera type')
        cameraType = 'Unknown'
    return cameraType

#%% wait until all channels acquired
ready = False
while not ready:
    channelFilters, rgbMode = getChannelsAcquired(localPath, rawPath)

    if len(channelFilters) == 0:
        print('No channels found in raw files. Waiting 1 minutes...')
        time.sleep(1 * 60)

    else:
        ready = True
        print('Channels found:', channelFilters)

cameraType = getCameraTypeOrientation(rawPath)

# %% [markdown]
# # compute time b/t first z planes of diff tiles

#%% use tqdm
# detect if software froze
def detectSoftwareFreeze(start, end, row, ii, jj, basePath, subject, startBody):
    if end - start > 15 * 60: # longer than 15 minutes per tile. Software likely froze
        # print('Tile', [str(row), str(ii), str(jj), '0'], 'taking too long. Software may have frozen. Sending email at', datetime.now())
        # sendEmailAlert(subject = subject, startBody = startBody, basePath = basePath)
        return True
    return False

# send email alert to user
def sendEmailAlert(subject, startBody, basePath):
    # send email alert
    sender = 'your-email-here'
    recipient = sender
    subject = subject
    email = EmailMessage()
    email["From"] = sender
    email["To"] = recipient
    email["Subject"] = subject

    # write body of email
    body = f'{startBody} at {str(datetime.now())} on current sample \n {basePath} \n Please restart. \n'
    email.set_content(body)

    # send the email
    # login to email
    smtp_server = "smtp-mail.outlook.com"
    port = 587
    print('Sending email alert to', recipient, 'at', str(datetime.now()))
    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()  # Secure the connection
            server.login(sender, "your-password-here") # password
            server.send_message(email)
        except Exception as e:
            print('Failed to send email alert:', e)

def refine_offset_3d(fixed_4d, moving_4d, initial_offset, channel=0, overlap_frac=0.15):
    """
    Perform translation refinement for the 'moving_4d' tile w.r.t. 'fixed_4d',
    using only the specified channel for cross-correlation.
    Both fixed_4d and moving_4d are (C, Z, Y, X).
    Returns an updated (z_off, y_off, x_off).
    """
    # Extract the single channel volumes:
    fixed_vol = fixed_4d[channel]   # shape: (Z, Y, X)
    moving_vol = moving_4d[channel] # shape: (Z, Y, X)

    # Decide how big the overlap is (e.g., ~15% of Y and X)
    Z, Y, X = fixed_vol.shape
    overlap_y = int(Y * overlap_frac)
    overlap_x = int(X * overlap_frac)
    
    # Example approach: align the right edge of fixed to the left edge of moving.
    # Overlap in Y is the last 'overlap_y' rows in fixed, first 'overlap_y' in moving.
    # Overlap in X is the last 'overlap_x' columns in fixed, first 'overlap_x' in moving.
    # This is just an example; adapt as needed for your tile layout.
    
    overlap_fixed = fixed_vol[:, -overlap_y:, -overlap_x:]   # shape: (Z, overlap_y, overlap_x)
    overlap_moving = moving_vol[:, :overlap_y, :overlap_x]   # shape: (Z, overlap_y, overlap_x)
    
    # Perform cross-correlation
    shift, error, phase = skimage.registration.phase_cross_correlation(overlap_fixed, overlap_moving)
    # shift is [delta_z, delta_y, delta_x]
    
    # Apply shift to the initial_offset
    refined = (
        initial_offset[0] + shift[0],
        initial_offset[1] + shift[1],
        initial_offset[2] + shift[2]
    )
    return refined

# stitch 3D volumes
async def stitch_3D_volumes(ptName, tileY, tileX, wavelengths, overlap_frac, ref_ch = 0):

    tifFiles = [f for f in rawTifPath.glob(f'{ptName}_*.tif')]

    rows = []
    for ii in range(tileY): # each row of tiles
        cols = []
        for jj in range(tileX): # each column of tiles

            chStack = []
            for kk, excite in enumerate(wavelengths): # each channel

                fileName = [f for f in tifFiles if f'_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_' in f.stem and 
                            str(excite) in f.stem]
                assert len(fileName) == 1
                zStack = tf.imread(fileName[0]) # ZYX
                chStack.append(zStack) # each ZYX
            chStack = np.stack(chStack, axis = 0) # CZYX
            cols.append(chStack)
        
        rows.append(cols)
        
    tiles = rows
    # Example check the shape of the first tile
    nC, Z_tile, Y_tile, X_tile = tiles[0][0].shape

    ##############################################################################
    # 3) Compute Nominal Offsets (Assuming Known Overlap)
    ##############################################################################

    # We'll assume purely translational offsets in Y, X, none in Z for each tile
    step_y = int(Y_tile * (1 - overlap_frac))  # ~85% of Y_tile
    step_x = int(X_tile * (1 - overlap_frac))  # ~85% of X_tile

    # offsets[r][c] -> (z_offset, y_offset, x_offset)
    offsets = [[(0,0,0) for _ in range(tileY)] for _ in range(tileX)]

    for r in range(tileY):
        for c in range(tileX):
            if r == 0 and c == 0:
                offsets[r][c] = (0, 0, 0)  # reference tile at origin
            else:
                z_off = 0
                y_off = r * step_y
                x_off = c * step_x
                offsets[r][c] = (z_off, y_off, x_off)

    ##############################################################################
    # 4) Refinement by Cross-Correlation (Only Using One Reference Channel)
    ##############################################################################
    # Example: refine each tile’s offset based on its left or top neighbor
    for r in range(tileY):
        for c in range(tileX):
            if r == 0 and c == 0:
                continue  # the reference tile, offset = (0,0,0)
            
            nominal_off = offsets[r][c]
            neighbors_refined = []
            
            # Refine vs left neighbor if c > 0
            if c > 0:
                fixed_tile = tiles[r][c-1]
                moving_tile = tiles[r][c]
                refined_off = refine_offset_3d(fixed_tile, moving_tile, nominal_off,
                                            channel=ref_ch, overlap_frac=overlap_frac)
                neighbors_refined.append(refined_off)
            
            # Refine vs top neighbor if r > 0
            if r > 0:
                fixed_tile = tiles[r-1][c]
                moving_tile = tiles[r][c]
                refined_off = refine_offset_3d(fixed_tile, moving_tile, nominal_off,
                                            channel=ref_ch, overlap_frac=overlap_frac)
                neighbors_refined.append(refined_off)
            
            # If we have two neighbors, we can average the offsets
            if len(neighbors_refined) == 1:
                offsets[r][c] = neighbors_refined[0]
            elif len(neighbors_refined) == 2:
                # average them
                z_avg = np.mean([off[0] for off in neighbors_refined])
                y_avg = np.mean([off[1] for off in neighbors_refined])
                x_avg = np.mean([off[2] for off in neighbors_refined])
                offsets[r][c] = (z_avg, y_avg, x_avg)

        ##############################################################################
        # 5) Determine Stitched Volume Size
        ##############################################################################

        all_offs = [offsets[r][c] for r in range(tileY) for c in range(tileX)]

        z_min = min(off[0] for off in all_offs)
        z_max = max(off[0] + Z_tile for off in all_offs)
        y_min = min(off[1] for off in all_offs)
        y_max = max(off[1] + Y_tile for off in all_offs)
        x_min = min(off[2] for off in all_offs)
        x_max = max(off[2] + X_tile for off in all_offs)

        z_min, z_max = int(z_min), int(z_max)
        y_min, y_max = int(y_min), int(y_max)
        x_min, x_max = int(x_min), int(x_max)

        Z_final = z_max - z_min
        Y_final = y_max - y_min
        X_final = x_max - x_min

        # Our final stitched volume is now (C, Z_final, Y_final, X_final)
        stitched_volume = np.zeros((nC, Z_final, Y_final, X_final), dtype=np.float32)
        count_volume   = np.zeros((Z_final, Y_final, X_final),      dtype=np.float32)
        # (We only need one "count_volume" because we do the same overlap count for all channels.)

        ##############################################################################
        # 6) Paste Tiles into Stitched Volume (Applying the Same Offset to All Channels)
        ##############################################################################
        for r in range(tileY):
            for c in range(tileX):
                tile_4d = tiles[r][c]        # shape: (nC, Z_tile, Y_tile, X_tile)
                z_off, y_off, x_off = offsets[r][c]
                
                # Convert offset to zero-based indices in the final volume
                z_start = int(z_off - z_min)
                y_start = int(y_off - y_min)
                x_start = int(x_off - x_min)
                
                z_end = z_start + Z_tile
                y_end = y_start + Y_tile
                x_end = x_start + X_tile
                
                # Accumulate each channel
                # stitched_volume has shape (nC, Z_final, Y_final, X_final)
                # tile_4d has shape (nC, Z_tile, Y_tile, X_tile)
                stitched_volume[:, z_start:z_end, y_start:y_end, x_start:x_end] += tile_4d
                count_volume[z_start:z_end, y_start:y_end, x_start:x_end] += 1

        # Now we handle dividing by `count_volume`. We do it per-channel in the stitched volume.
        non_zero = count_volume > 0
        for c in range(nC):
            stitched_slice = stitched_volume[c]
            stitched_slice[non_zero] /= count_volume[non_zero]

        # # visualize
        # napari.view_image(stitched_volume, channel_axis=0, blending = 'additive', contrast_limits=[0, 30000])

        ##############################################################################
        # 7) Save the Multi-Channel Stitched Result
        ##############################################################################
        # The final result is shape (nC, Z_final, Y_final, X_final).
        # Depending on your tools, you might want to transpose it or keep it in (C, Z, Y, X).
        fileOut = volumeStitchPath / f'{ptName}_stitched.tif'
        tf.imwrite(fileOut, stitched_volume.astype(np.float32))

# reformat Linux home paths to Windows
def reformatPathLinuxToWindows(linuxPath):

    # # check OS
    # if platform.system() == 'Linux':
    #     # Linux has different parent path than windows. Reformat parent to match windows so user can run Fiji in windows
    #     path = linuxPath.parts
    #     # find Desktop string match
    #     for idxCoskun, part in enumerate(path):
    #         if part == 'Desktop':
    #             break
    #     if 'coskun-lab' in path:
    #         baseFolder = 'coskun-lab'
    #     elif 'coskun-lab2' in path:
    #         baseFolder = 'coskun-lab2'
    #     else:
    #         raise ValueError('Unknown base folder')
    #     # add first half of path
    #     path = Path('Y:', baseFolder, *path[idxCoskun + 1:]).parts
    #     # remove duplicate entries
    #     ordered = []
    #     for part in path:
    #         if part not in ordered:
    #             ordered.append(part)
    #     windowsPath = Path(*ordered)

    # elif platform.system() == 'Windows':
    #     windowsPath = linuxPath # no reformatting needed

    # else:
    #     raise ValueError('Unknown OS')
    windowsPath = linuxPath

    return windowsPath

#%% write processing command to text file and save to disk for user to run later
def writeProcessCommandTxt(rawPath, overlapFrac, dimZ, tileY, tileX, nRepFrames, numCoords, stitchMode, localPath):
    txtFile = rawPath.parent / 'process_images_command.txt'
    f = open(txtFile, 'w')
    # get current absolute path of running scipt
    currScriptPath = Path(os.path.abspath(__file__))
    # write linux command
    if platform.system() == 'Linux':
        f.write('\n Linux command \n')
        f.write(f'python3 "{currScriptPath}" {urllib.parse.quote(str(rawPath))} {overlapFrac} {dimZ} {tileY} {tileX} {nRepFrames} {numCoords} {stitchMode} {localPath}')
        f.write('\n\n Windows command \n')
        # reformat Linux path to Windows
        windowsPath = Path(r'Y:\coskun-lab\Nicky\02 Microscope Tools\Cephla Squid\octopi-research-master\Current\software\control\process_images_real_time_interactive.py')
        f.write(f'python "{str(windowsPath)}" \n')
        f.write(f'{str(reformatPathLinuxToWindows(basePath))} \n')
        
    # elif platform.system() == 'Windows':
    #     # reformat windows path to linux

    #     f.write('\n Linux command \n')
    #     f.write(f'python3 "{currScriptPath}" {urllib.parse.quote(str(rawPath))} {overlapFrac} {dimZ} {tileY} {tileX} {nRepFrames} {numCoords} {stitchMode} {localPath}')
    #     # write windows command
    #     f.write('\n\n Windows command \n')
    #     f.write(f'python process_images_real_time.py {urllib.parse.quote(str(rawPath))} {overlapFrac} {dimZ} {tileY} {tileX} {nRepFrames} {numCoords} {stitchMode} ""')

    f.close()
    
# writeProcessCommandTxt(rawPath, overlapFrac, dimZ, tileY, tileX, nRepFrames, numCoords, stitchMode, localPath) # skip 

#%% process images

# find acquired files for given tile
def locateTileFiles(rawFiles, row, ii, jj, dimZ, channelFilters, start, basePath, nRepFrames):

    # find last z plane for first channel
    # print('Searching', [str(row), str(ii), str(jj), '0'])
    fileName = [f for f in rawFiles if [str(row), str(ii), str(jj)] == f.stem.split('_')[:3]]

    # remove files with duplicate stems but different parents
    # Group files by their stem
    files_by_stem = defaultdict(list)
    for file in fileName:
        files_by_stem[file.stem].append(file)

    if len(files_by_stem) != dimZ * len(channelFilters) * nRepFrames: # not found
        # expand search raw files
        # rawFiles = [f for f in rawPath.glob('*.tiff')]
        rawFiles = getRawFiles(localPath, rawPath)
        fileName = [f for f in rawFiles if [str(row), str(ii), str(jj)] == f.stem.split('_')[:3]]
    
    else: # successfully found file
        pass

    assert len(fileName) == dimZ * len(channelFilters) * nRepFrames, 'Missing channel files'

    # if not frozen:
    #     frozen = detectSoftwareFreeze(start, end, row, ii, jj, basePath, 
    #                                     subject = "Detected Squid confocal software crash", 
    #                                     startBody = "Squid software crashed")

    return rawFiles

# export raw TIFF z stack
def combineZStack(row, ii, jj, dimZ, channelFilters, 
                  networkFiles, rawPath, rawTifPath, ptName, 
                  useCoordNames, nRepFrames):
    # get all files for this channel
    chFiles = [f for f in networkFiles if f.stem.startswith(f'{str(row)}_{str(ii)}_{str(jj)}_')]
    if len(chFiles) < len(channelFilters) * dimZ: # files not found
        # files not found. Keep searching network folder
        networkFiles = [f for f in rawPath.glob('*.tiff')]
        chFiles = [f for f in networkFiles if f.stem.startswith(f'{str(row)}_{str(ii)}_{str(jj)}_')]
    assert len(chFiles) == len(channelFilters) * dimZ * nRepFrames, 'Missing channel files'

    chStack = []
    for ll, excite in enumerate(channelFilters): # each channel

        # find z stack files
        zStackFiles = [f for f in chFiles if 
                        f.stem.startswith(f'{str(row)}_{str(ii)}_{str(jj)}_') and 
                        str(excite) + '.' in f.name]
        assert len(zStackFiles) == dimZ * nRepFrames, 'Missing z stack files'
        
        zStack = []
        for kk in range(dimZ): # each z plane

            # read plane with dask
            fileName = [f for f in zStackFiles if [str(row), str(ii), str(jj), str(kk)] == f.stem.split('_')[:4] and \
                    '_' + str(excite) + '.' in f.name]
            assert len(fileName) == nRepFrames

            # if multiple frames (SRRF) then average them
            if nRepFrames > 1:
                
                # read all frames
                frameStack = []
                for frame in fileName:
                    # plane = lazy_read(frame)
                    # plane = da.from_delayed(plane, shape=[dimY, dimX], dtype=dtype)
                    # plane = tf.imread(frame)
                    plane = dask_image.imread.imread(frame).squeeze() # YX
                    frameStack.append(plane) # each YX

                # stack all frames
                frameStack = np.stack(frameStack, axis=0) # TYX
                # average all frames for visual purposes
                plane = np.mean(frameStack, axis=0) # YX
                
            else: # single frame, normal acquisition
                # plane = lazy_read(fileName[0])
                # plane = da.from_delayed(plane, shape=[dimY, dimX], dtype=dtype)
                # plane = tf.imread(fileName[0]) # YX
                plane = dask_image.imread.imread(fileName[0]).squeeze() # YX

            zStack.append(plane) # each YX
            
        zStack = np.stack(zStack, axis=0) # ZYX single channel

        ################## run deconvolution? ##################

        ################## run flat field correction ##################
        # only run flat field on non focus channels if using well plate multipoint
        if ll > 0 and useCoordNames:
            zStack = flatFieldCorrection(zStack)

        chStack.append(zStack) # each ZYX

        # also create TIF stack for each channel
        fileOut = f'{ptName}_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_{excite}.tif'
        fileOut = rawTifPath / fileOut
        tf.imwrite(fileOut, data=zStack, photometric = 'minisblack', bigtiff=True)           

    chStack = np.stack(chStack, axis=0) # CZYX, single tile
    return chStack # CZYX dask array

lazy_background = dask.delayed(skimage.restoration.rolling_ball)

# flat field correction
def flatFieldCorrection(zStack):
    if isinstance(zStack, da.Array):
        rawStack = zStack.compute() # compute into RAM

    # fit flatfield and darkfield
    basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)
    basic.fit(rawStack)
    # correct original image
    correctZStack = basic.transform(rawStack)

    # return as dask array if original was dask array
    if isinstance(zStack, da.Array):
        correctZStack = da.from_array(correctZStack, chunks = zStack.chunks)
    
    return correctZStack

# compute 2D tiles for stitching
def reformatTiles(stitchMode, chStack, row, ii, jj, fileNum, channelFilters, 
                  rawTilePath, processedTilePath, renderPath, 
                  ptName, useCoordNames, volumeTilePath, 
                  rgbMode):
    if rgbMode:
        # move channel axis to first
        chStack = np.moveaxis(chStack, -1, 0) # CZYX
    if stitchMode == 'MIP':
        mip = np.max(chStack, axis = 1) # CYX

    elif stitchMode == 'BestFocus':
        # compute best focus plane instead of MIP
        idxFocus, peaks = findFocusLapVar(chStack[0, ...].compute()) # ZYX
        # if multiple peaks found, choose closest to middle 
        if len(peaks) > 1:
            # print('Multiple fixed peaks found...')
            middle = chStack.shape[1] // 2
            diff = np.abs(peaks - middle)
            closest = np.argmin(diff)
            idxFocus = peaks[closest]
        mip = chStack[:, idxFocus + 1, :, :] # CYX. best focus plane

    else:
        raise ValueError('Unknown stitch mode')
    
    if rgbMode: # RGB
        # move channel axis back to last
        mip = np.moveaxis(mip, 0, -1).compute() # YXC
        # export raw TIF
        fileOut = f'tile_{str(fileNum).zfill(5)}.tif'
        fileOut = Path(rawTilePath, ptName, fileOut)
        if not fileOut.parent.exists():
            fileOut.parent.mkdir(parents = True)
        tf.imwrite(fileOut, mip, photometric = 'rgb')

        # render MIP PNG for focus check
        fig, ax = plt.subplots(dpi = 300, facecolor = 'black')
        ax.imshow(mip)
        ax.set_title(f'{str(row)} tile {str(fileNum).zfill(5)} RGB {stitchMode}', color = 'white')
        fileOut = f'{str(row)}_{str(ptName)}_tile_{str(fileNum).zfill(5)}_Y{str(ii).zfill(3)}_X{str(jj).zfill(3)}_RGB.png' if useCoordNames \
            else f'{str(row)}_tile_{str(fileNum).zfill(5)}_Y{str(ii).zfill(3)}_X{str(jj).zfill(3)}_RGB.png'
        fileOut = Path(renderPath, fileOut)
        plt.savefig(fileOut, bbox_inches = 'tight', dpi = 'figure', pad_inches = 0)
        plt.clf() # clear figure
        plt.close()
        return # skip rest
    
    # export channels separately
    procMip = []
    for kk, excite in enumerate(channelFilters): # each channel
        planeOut = mip[kk, ...] # YX

        # subtract background
        if useCoordNames: # well plate multipoint
            procPlaneOut = planeOut # no background subtraction if flat field correction was done
        else: # large tile scan
            background = lazy_background(planeOut, radius = 50)
            background = da.from_delayed(background, shape = planeOut.shape, dtype = planeOut.dtype)
            procPlaneOut = planeOut - background
        procMip.append(procPlaneOut) # each YX

    # compute MIPs into RAM
    mip = mip.compute() # CYX raw
    procMip = np.stack(procMip, axis = 0).compute() # CYX processed

    # export raw and processed channels
    for kk, excite in enumerate(channelFilters): # each channel
        # export raw TIF
        planeOut = mip[kk, ...] # YX
        fileOut = f'tile_{str(fileNum).zfill(5)}_CH{kk + 1}.tif'
        fileOut = Path(rawTilePath, ptName, fileOut)
        if not fileOut.parent.exists():
            fileOut.parent.mkdir(parents = True)
        tf.imwrite(fileOut, planeOut, photometric = 'minisblack')

        # export processed TIF
        procPlaneOut = procMip[kk, ...] # YX
        fileOut = f'tile_{str(fileNum).zfill(5)}_CH{kk + 1}.tif'
        fileOut = Path(processedTilePath, ptName, fileOut)
        if not fileOut.parent.exists():
            fileOut.parent.mkdir(parents = True)
        tf.imwrite(fileOut, procPlaneOut, photometric = 'minisblack')

    # also compute DAPI MIP render to confirm focus plane
    focusMip = mip[0, ...] # YX DAPI only
    fig, ax = plt.subplots(dpi = 300, facecolor = 'black')
    ax.imshow(focusMip, cmap = 'coolwarm')
    ax.set_title(f'{str(row)} tile {str(fileNum).zfill(5)} {channelFilters[0]} {stitchMode}', color = 'white')
    fileOut = f'{str(row)}_{str(ptName)}_tile_{str(fileNum).zfill(5)}_Y{str(ii).zfill(3)}_X{str(jj).zfill(3)}_{str(channelFilters[0])}.png' if useCoordNames \
        else f'{str(row)}_tile_{str(fileNum).zfill(5)}_Y{str(ii).zfill(3)}_X{str(jj).zfill(3)}_{str(channelFilters[0])}.png'
    fileOut = Path(renderPath, fileOut)
    plt.savefig(fileOut, bbox_inches = 'tight', dpi = 'figure', pad_inches = 0)
    plt.clf() # clear figure
    plt.close()

    ###################### reformat tile for 3D stitching if using multipt ######################
    if useCoordNames:
        chStack = np.swapaxes(chStack, 0, 1) # CZYX -> ZCYX
        fileOut = f'tile_{str(fileNum).zfill(5)}.tif'
        fileOut = volumeTilePath / ptName / fileOut
        if not fileOut.parent.exists():
            fileOut.parent.mkdir(parents = True)
        tf.imwrite(fileOut, chStack.compute(), photometric = 'minisblack', imagej = True)

# merge RGB
def mergeRGB(rawTifPath, ptName, ii, jj):

    # merge RGB
    R = [f for f in rawTifPath.glob('*.tif') if 'BF_LED_matrix_full_R' in f.stem and f.stem.startswith(f'{ptName}_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_')]
    assert len(R) == 1
    R = tf.imread(R[0])
    G = [f for f in rawTifPath.glob('*.tif') if 'BF_LED_matrix_full_G' in f.stem and f.stem.startswith(f'{ptName}_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_')]
    assert len(G) == 1
    G = tf.imread(G[0])
    B = [f for f in rawTifPath.glob('*.tif') if 'BF_LED_matrix_full_B' in f.stem and f.stem.startswith(f'{ptName}_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_')]
    assert len(B) == 1
    B = tf.imread(B[0])
    # merge RGB
    # gray = np.zeros_like(R)
    # RGB = skimage.color.gray2rgb(gray)
    # RGB[..., 0] = R
    # RGB[..., 1] = G
    # RGB[..., 2] = B
    # RGB = np.stack([R, G, B], axis=0)
    RGB = cv2.merge([R, G, B])
    # xGroup.create_dataset('BF_LED_matrix_full_RGB', data=RGB)
    # also save as TIF separately
    # fileOut = 'Point' + str(row).zfill(3) + '_y' + str(ii).zfill(3) + '_x' + str(jj).zfill(3) + '_RGB.tif'
    fileOut = f'{ptName}_y{str(ii).zfill(3)}_x{str(jj).zfill(3)}_RGB.tif'
    fileOut = rawTifPath / fileOut
    tf.imwrite(fileOut, photometric = 'rgb', data=RGB)
    return RGB # ZYXC

# pause script if RAM usage is too high
def pauseCheckRAM():
    # temp pause if RAM usage is too high
    while psutil.virtual_memory().percent > 65:
        print('RAM usage too high. Pausing for 1 minute...')
        time.sleep(60)

#%% write stitching Fiji macro ahead of time to run later (optional)
# reformat path for Fiji Macro
def reformatPathFiji(path):
    return str(path).replace('\\', '/')

# stitch with MIST
def stitch_2D_MIST(macro, jj, excite, folderOut, tileX, tileY, 
                   overlapFrac, rawTilePath, processedTilePath, cameraType,
                   rgbMode = False):
    if jj == 0: 
        # compute stitching on raw images
        macro.write('run("MIST", "gridwidth=')
        macro.write(str(tileX))
        macro.write(' gridheight=')
        macro.write(str(tileY))
        macro.write(' starttile=1 imagedir=[')
        splitFolder = str(folderOut.stem)
        splitFolder = reformatPathLinuxToWindows(rawTilePath) / splitFolder
        splitFolder.mkdir(exist_ok = True, parents=True)
        splitFolder = reformatPathFiji(splitFolder)
        macro.write(str(splitFolder))
        macro.write('] filenamepattern=tile_{ppppp}_CH1.tif filenamepatterntype=SEQUENTIAL gridorigin=')
        if cameraType == 'Hamamatsu':
            macro.write('LR')
        elif cameraType == 'Sony':
            macro.write('UR')
        macro.write(' assemblefrommetadata=false assemblenooverlap=false globalpositionsfile=[] numberingpattern=HORIZONTALCONTINUOUS startrow=0 startcol=0 extentwidth=')
        macro.write(str(tileX))
        macro.write(' extentheight=')
        macro.write(str(tileY))
        macro.write(' timeslices=0 istimeslicesenabled=false outputpath=[')
        macro.write(reformatPathFiji(folderOut))
        macro.write('] displaystitching=false outputfullimage=true outputmeta=true outputimgpyramid=false blendingmode=LINEAR blendingalpha=1.5 compressionmode=UNCOMPRESSED outfileprefix=')
        macro.write(f'{excite}_')
        macro.write('img_t1_z')
        macro.write(str(jj + 1))
        macro.write('_c1_ unit=MICROMETER unitx=1.0 unity=1.0 programtype=AUTO numcputhreads=32 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw\\fftPlans fftwlibrarypath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw stagerepeatability=0 horizontaloverlap=')
        macro.write(str(overlapFrac * 100))
        macro.write(' verticaloverlap=')
        macro.write(str(overlapFrac * 100))
        macro.write(' numfftpeaks=0 overlapuncertainty=NaN isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=true isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE"); \n')

    # else: 
    # apply stitching to other processed channels
    macro.write('run("MIST", "gridwidth=')
    macro.write(str(tileX))
    macro.write(' gridheight=')
    macro.write(str(tileY))
    macro.write(' starttile=1 imagedir=[')
    splitFolder = str(folderOut.stem)
    splitFolder = reformatPathLinuxToWindows(processedTilePath) / splitFolder if not rgbMode else reformatPathLinuxToWindows(rawTilePath) / splitFolder
    # splitFolder = reformatPathLinuxToWindows(processedTilePath) / splitFolder
    splitFolder.mkdir(exist_ok = True, parents=True)
    splitFolder = reformatPathFiji(splitFolder)
    macro.write(str(splitFolder))
    macro.write('] filenamepattern=tile_{ppppp}_CH')
    macro.write(str(jj + 1))
    macro.write('.tif filenamepatterntype=SEQUENTIAL gridorigin=')
    if cameraType == 'Hamamatsu':
        macro.write('LR')
    elif cameraType == 'Sony':
        macro.write('UR')
    macro.write(' assemblefrommetadata=true assemblenooverlap=false globalpositionsfile=[')
    macro.write(reformatPathFiji(folderOut / f'{channelFilters[0]}_img_t1_z1_c1_global-positions-0.txt'))
    macro.write('] numberingpattern=HORIZONTALCONTINUOUS startrow=0 startcol=0 extentwidth=')
    macro.write(str(tileX))
    macro.write(' extentheight=')
    macro.write(str(tileY))
    macro.write(' timeslices=0 istimeslicesenabled=false outputpath=[')
    macro.write(reformatPathFiji(folderOut))
    macro.write('] displaystitching=false outputfullimage=true outputmeta=true outputimgpyramid=false blendingmode=LINEAR blendingalpha=1.5 compressionmode=UNCOMPRESSED outfileprefix=')
    macro.write(f'{excite}_')
    macro.write('img_t1_z')
    macro.write(str(jj + 1))
    macro.write('_c1_ unit=MICROMETER unitx=1.0 unity=1.0 programtype=AUTO numcputhreads=32 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw\\fftPlans fftwlibrarypath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw stagerepeatability=0 horizontaloverlap=')
    macro.write(str(overlapFrac * 100))
    macro.write(' verticaloverlap=')
    macro.write(str(overlapFrac * 100))
    macro.write(' numfftpeaks=0 overlapuncertainty=NaN isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=true isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE"); \n')

'''
filelist = getFileList("Y:/coskun-lab/Nicky/49 Cystic Fibrosis - Rabin/Data/24 well plate 003A/13Feb2025 cycle 1 HCR and IF/07 stitched 3D volumes/A1-1") 
dimZ = 0;
for (i = 0; i < lengthOf(filelist); i++) {
    if (!endsWith(filelist[i], ".tif")) { 
//        open(directory + File.separator + filelist[i]);
		print(filelist[i]);
		dimZ = dimZ + 1;
}
}
dimZ = floor(dimZ / 5);
print(dimZ);

run("Bio-Formats Importer", "open=[Y:/coskun-lab/Nicky/49 Cystic Fibrosis - Rabin/Data/24 well plate 003A/13Feb2025 cycle 1 HCR and IF/07 stitched 3D volumes/A1-1/img_t1_z01_c1] color_mode=Default group_files rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack dimensions axis_1_number_of_images=52 axis_1_axis_first_image=1 axis_1_axis_increment=1 axis_2_number_of_images=5 axis_2_axis_first_image=1 axis_2_axis_increment=1 contains=[] name=[Y:/coskun-lab/Nicky/49 Cystic Fibrosis - Rabin/Data/24 well plate 003A/13Feb2025 cycle 1 HCR and IF/07 stitched 3D volumes/A1-1/img_t1_z<01-52>_c<1-5>]");

run("Split Channels");
selectImage("C1-imgStack.tif");

saveAs("Tiff", "Y:/coskun-lab/Nicky/07 Temp/Fiji export channels/A1-1_Fluorescence_405_nm.tif");

'''
# stack volume planes directly in Fiji
def stack_volume_planes_Fiji(macro, volOutPath, channelFilters, ptName):
    # stack volume planes directly in Fiji macro
    macro.write('// Stack volume planes after 3D stitching \n')
    # get Z dimensions
    macro.write('filelist = getFileList("')
    macro.write(reformatPathFiji(volOutPath))
    macro.write('");\n')
    macro.write('dimZ = 0; \n')
    macro.write('for (i = 0; i < lengthOf(filelist); i++) {\n')
    macro.write('    if (!endsWith(filelist[i], ".tif")) { \n')
    macro.write('        dimZ = dimZ + 1; \n')
    macro.write('    }\n')
    macro.write('}\n')
    macro.write(f'dimZ = floor(dimZ / {len(channelFilters)}); \n')

    # open combined stack in Fiji
    macro.write('run("Bio-Formats Importer", "open=[')
    macro.write(f'{reformatPathFiji(volOutPath / "img_t1_z01_c1")}')
    macro.write('] color_mode=Default group_files rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT use_virtual_stack dimensions axis_1_number_of_images=')
    macro.write(str(dimZ))
    macro.write(' axis_1_axis_first_image=1 axis_1_axis_increment=1 axis_2_number_of_images=')
    macro.write(str(len(channelFilters)))
    macro.write(' axis_2_axis_first_image=1 axis_2_axis_increment=1 contains=[] name=[')
    macro.write(f'{reformatPathFiji(volOutPath / f"img_t1_z<01-{dimZ}>_c<1-{len(channelFilters)}>")}')
    macro.write(']"); \n')
    # Bio-Formats doesn't work. Open each plane instead

    macro.write('rename("imgStack"); \n')
    # split channels
    macro.write('run("Split Channels"); \n')
    # save channels separately
    for jj, excite in enumerate(channelFilters): # each channel
        macro.write(f'selectImage("C{jj + 1}-imgStack"); \n')
        macro.write('saveAs("Tiff", "')
        macro.write(reformatPathFiji(volOutPath / f"{ptName}_{excite}.tif"))
        macro.write('"); \n')
        macro.write('close; \n')
    macro.write('run("Close All"); \n') # close all images

# stitch with Grid Collection
def stitch_3D_Grid_Collection(macro, tileX, tileY, overlapFrac, cameraType, tilePath, stitchPath, ptName, channelFilters, rgbMode = False):

    # run stitching command
    macro.write('run("Grid/Collection stitching",')
    # macro.write(' "type=[Filename defined position]')
    # macro.write(' order=[Defined by filename         ]') # may need to change
    macro.write(' "type=[Grid: snake by rows]')
    if cameraType == 'Hamamatsu':
        macro.write(' order=[Left & Up]')
    elif cameraType == 'Sony':
        macro.write(' order=[Left & Down]') # may need to change
    macro.write(' grid_size_x=' + str(tileX)) # num of X tiles
    macro.write(' grid_size_y=' + str(tileY)) # num of Y tiles
    macro.write(' tile_overlap=' + str(int(overlapFrac * 100))) # overlap fraction percent
    macro.write(' first_file_index_i=1')
    folderOut = Path(tilePath, ptName)
    splitFolder = str(folderOut.stem)
    splitFolder = reformatPathLinuxToWindows(tilePath) / splitFolder
    if platform.system() == 'Windows': splitFolder.mkdir(exist_ok = True, parents=True)
    splitFolder = reformatPathFiji(splitFolder)
    macro.write(' directory=[' + splitFolder + '] file_names=tile_{iiiii}.tif')
    macro.write(' output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50')
    # macro.write(' compute_overlap') # sometimes messes up stitching
    macro.write(' computation_parameters=[Save memory (but be slower)] image_output=[Write to disk]')
    volOutPath = reformatPathLinuxToWindows(Path(stitchPath, ptName))
    if platform.system() == 'Windows': volOutPath.mkdir(exist_ok = True)
    macro.write(' output_directory=[' + reformatPathFiji(volOutPath) + ']"); \n')

    # # stack volume planes directly in Fiji macro
    # stack_volume_planes_Fiji(macro, volOutPath, channelFilters, ptName)

if RANK == 0 and useCoordNames: # only run on master node. 3D stitching macro
    # write Fiji macro
    now = datetime.now() # current date and time
    date_time = now.strftime("%d%b%Y")
    macroName = f'{date_time}_stitch_Rank{str(numCoords)}.ijm'
    macroName = largeImagePath / macroName
    macro = open(macroName, 'w')

    # add run macro line as comment in first line of actual macro
    runCommand = reformatPathLinuxToWindows(linuxPath=largeImagePath)
    # runCommand = largeImagePath
    runCommand = runCommand / macroName.name
    macro.write('// runMacro("' + reformatPathFiji(runCommand) + '"); \n\n')

    # split Fiji macro for HPC
    macro.write('rank = getArgument(); \n')
    macro.write('rank = parseInt(rank); \n\n')

    for row in range(numCoords): # each multipt. Only one for large scan

        ptName = f'Point{str(row).zfill(3)}'
        if useCoordNames and pd.notna(coords['ID'].iloc[row]):
            ptName = coords.ID.iloc[row]

        if useCoordNames:
            folderOut = largeImagePath / str(coords['ID'].iloc[row] if pd.notna(coords['ID'].iloc[row]) else ptName)
            # create 3D folders
            Path(volumeTilePath / folderOut.stem).mkdir(exist_ok = True)
            Path(volumeStitchPath / folderOut.stem).mkdir(exist_ok = True)
        else:
            folderOut = largeImagePath / ptName
        folderOut.mkdir(exist_ok = True)   

        macro.write(f'// {ptName} \n')
        macro.write(f'if (rank == {row + 1}) ')
        macro.write('{ \n')

        # reformat linux to windows
        folderOut = reformatPathLinuxToWindows(folderOut)

        # use MIST to stitch large scan
        for jj, excite in enumerate(channelFilters): # each channel
            # stitch with MIST
            stitch_2D_MIST(macro, jj, excite, folderOut, tileX, 
                        tileY, overlapFrac, rawTilePath, processedTilePath, cameraType,
                        rgbMode=rgbMode)

        if useCoordNames: # multipt 3D volume stitching
            # use Grid/Collection stitching
            stitch_3D_Grid_Collection(macro, tileX, tileY, overlapFrac, cameraType, volumeTilePath, volumeStitchPath, folderOut.stem, channelFilters)

        macro.write('} \n\n')

    # quit Fiji when finished to save RAM
    # macro.write('run("Quit"); \n')
    macro.close()

    # print out command to run this macro in Fiji
    print('runMacro("' + reformatPathFiji(runCommand)+ '");')

if onlyWriteMacro: sys.exit('Only writing macro. Exiting...')

#%% Processing mode
totalIter = numCoords * tileY * tileX

lazy_read = dask.delayed(tf.imread)

# rawFiles = [f for f in rawPath.glob('*.tiff')]
rawFiles = getRawFiles(localPath, rawPath)
networkFiles = [f for f in rawPath.glob('*.tiff')]
print('Sample', str(basePath))

#### give user option to continue processing or only move files
# onlyMoveFiles, timedOut = timedInput('Only moves files (m) or process images on this computer (p)?:', timeout = 5 * 60)
onlyMoveFiles = 'p' # process all files on cluster
# if timedOut:
#     if platform.system() == 'Linux':
#         onlyMoveFiles = 'm' # default only move files on Linux
#         print('Timed out. Only moving files...')
#     elif platform.system() == 'Windows':
#         onlyMoveFiles = 'p' # default process files on Windows
#         print('Timed out. Processing all files...')
#     else: 
#         raise ValueError('Unknown OS')

#%% simultaneous processing w/ acquisition
# HPC processing
def process_single_tile_multipt(index, row1, useCoordNames, tileY, tileX, 
                        dimZ, channelFilters, nRepFrames, rawFiles, 
                        localPath, rawPath, rawTifPath, 
                        processedTilePath, renderPath, 
                        volumeTilePath, volumeStitchPath):

    ptName = f'Point{str(index).zfill(3)}'
    if useCoordNames and pd.notna(coords['ID'].iloc[index]):
        ptName = coords.ID.iloc[index]

    xScanDir = 1 # snake by indexs
    fileNum = 0 # each frame step
    for ii in range(tileY): # each Y tile

        for jj in range(tileX): # each X tile

            # time each tile. If takes too long then send email to user
            start = time.time()
            
            # compute x tile index based on scan direction
            jj = jj if xScanDir == 1 else tileX - 1 - jj

            # locate tile files
            rawFiles = locateTileFiles(rawFiles, index, ii, jj, dimZ, channelFilters, start, basePath, nRepFrames)
    
            # print('Found', [str(index), str(ii), str(jj), '0'])
            ##### move files to network
            copyLocalFilesToNetwork(localPath, rawPath, ii, jj, index)

            # only move files or do all processing?
            if onlyMoveFiles == 'm':
                continue
            
            #### export raw TIFF z stack
            chStack = combineZStack(index, ii, jj, dimZ, channelFilters, 
                                    networkFiles, rawPath, rawTifPath, ptName, 
                                    useCoordNames, nRepFrames) # CZYX unless RGB
            
            ##### merge RGB if applicable
            if rgbMode: chStack = mergeRGB(rawTifPath, ptName, ii, jj) # YX3

            #### compute tiles for stitching
            fileNum += 1
            reformatTiles(stitchMode, chStack, index, ii, jj, fileNum, channelFilters, rawTilePath, 
                            processedTilePath, renderPath, ptName, useCoordNames, 
                            volumeTilePath = volumeTilePath if useCoordNames else None, 
                            rgbMode = rgbMode)
            
            # record tile dimensions for later
            _, _, yTileDim, xTileDim = chStack.shape # CZYX
            del chStack # save RAM

# HPC processing
def process_single_tile_large_scan(ptName, useCoordNames, ii, jj, tileX, 
                        dimZ, channelFilters, nRepFrames, rawFiles, 
                        localPath, rawPath, rawTifPath, 
                        processedTilePath, renderPath, 
                        volumeTilePath, coordIdx):
    # time each tile. If takes too long then send email to user
    start = time.time()

    # Define xScanDir as a function of ii
    xScanDir = 1 if ii % 2 == 0 else -1
    
    # compute x tile coordIdx based on scan direction
    jj = jj if xScanDir == 1 else tileX - 1 - jj

    # locate tile files
    rawFiles = locateTileFiles(rawFiles, coordIdx, ii, jj, dimZ, channelFilters, start, basePath, nRepFrames)

    # print('Found', [str(coordIdx), str(ii), str(jj), '0'])
    ##### move files to network
    copyLocalFilesToNetwork(localPath, rawPath, ii, jj, coordIdx)

    # only move files or do all processing?
    if onlyMoveFiles == 'm':
        return
    
    #### export raw TIFF z stack
    chStack = combineZStack(coordIdx, ii, jj, dimZ, channelFilters, 
                            networkFiles, rawPath, rawTifPath, ptName, 
                            useCoordNames, nRepFrames) # CZYX unless RGB
    
    ##### merge RGB if applicable
    if rgbMode: chStack = mergeRGB(rawTifPath, ptName, ii, jj) # YX3

    #### compute tiles for stitching
    fileNum = ii * tileX + jj + 1
    reformatTiles(stitchMode, chStack, coordIdx, ii, jj, fileNum, channelFilters, rawTilePath, 
                    processedTilePath, renderPath, ptName, useCoordNames, 
                    volumeTilePath = volumeTilePath if useCoordNames else None, 
                    rgbMode = rgbMode)
    
    # # record tile dimensions for later
    # _, _, yTileDim, xTileDim = chStack.shape # CZYX
    del chStack # save RAM

if useCoordNames: # multipt sample (e.g. well plate)
    # --- Problem Setup & DataFrame Creation ---
    # In a real scenario, Rank 0 might load data and broadcast/scatter,
    # or all ranks might load data if feasible.
    # For simplicity here, all ranks create the same DataFrame.
    # N = 100 # Number of rows in the DataFrame
    num_rows = len(coords) # Use actual DataFrame length

    # --- Work Distribution (Distribute Row Indices) ---
    # Divide the total rows (num_rows) as evenly as possible among 'size' processes.
    # items_per_process = num_rows // SIZE
    # remainder = num_rows % SIZE

    # start_index = RANK * items_per_process + min(RANK, remainder)
    # end_index = start_index + items_per_process + (1 if RANK < remainder else 0)

    # --- Get Local DataFrame Slice ---
    # Each rank selects its portion of the DataFrame using iloc (integer-location based)
    # This is efficient as it avoids iterating over rows the rank won't process.
    # local_df_slice = coords.iloc[start_index:end_index]
    local_df_slice = coords.iloc[np.arange(RANK, num_rows, SIZE)] # each rank gets its own slice of the DataFrame

    local_results = [] # Results computed by this specific rank

    if RANK == 0:
        print(f"Starting parallel computation with {SIZE} processes...")
        print(f"Total coords: {num_rows}")
        print(f"Rank {RANK}: Processing coords: {local_df_slice.index.tolist()}") # -1 for display

    # --- Parallel Computation using itertuples ---
    # Iterate over the *local slice* of the DataFrame.
    # itertuples() is generally faster than iterrows().
    # index=True (default) includes the DataFrame index as the first element (row.Index).
    # name='Pandas' (default) sets the name of the namedtuples yielded.
    for row in local_df_slice.itertuples(index=True):
        # 'row' is a namedtuple like Pandas(Index=..., A=..., B=..., C=...)
        original_index = row.Index # Access the original DataFrame index
        print(f"Rank {RANK}: Processing coord {original_index}")
        # Pass the index and the whole row tuple to the calculation function
        res = process_single_tile_multipt(original_index, row, 
                            useCoordNames, tileY, tileX,
                            dimZ, channelFilters, nRepFrames, rawFiles,
                            localPath, rawPath, rawTifPath,
                            processedTilePath, renderPath,
                            volumeTilePath, volumeStitchPath)
        local_results.append(res)

    print(f"Rank {RANK}: Finished its computations. Processed {len(local_df_slice)} rows. Found {len(local_results)} results.")

    # --- Result Aggregation (Gather) ---
    # Collect the results from all processes onto the root process (rank 0).
    gathered_results = COMM.gather(local_results, root=0) # CAN BE VERY SLOW

    # --- Final Output/Processing (on root rank) ---
    if RANK == 0:
        print("Gathering results on rank 0...")
        # The gathered_results will be a list of lists. Flatten it.
        final_results = []
        if gathered_results: # Ensure gathered_results is not None
            for sublist in gathered_results:
                final_results.extend(sublist)

        print(f"Finished parallel computation. Final result length: {len(final_results)}")

        # --- Verification (Example) ---
        if len(final_results) == num_rows:
            print(f"Result length matches DataFrame rows ({num_rows}). Further content verification might be needed.")
            # Optional: Compare against serial execution if feasible
            # serial_results = []
            # for row in df.itertuples(index=True):
            #   serial_results.append(process_single_tile(row.Index, row))
            # if sorted(final_results) == sorted(serial_results): # Sorting assumes order might differ slightly depending on gather
            #   print("Verification successful (results match serial execution).")
            # else:
            #   print("Verification FAILED! Content mismatch detected.")
        else:
            print(f"Verification FAILED! Final result length ({len(final_results)}) does not match coords ({num_rows}).")

else: # large tile scan. useCoordNames = False
    local_results = []  # List to store results generated by this rank
    total_tiles = tileY * tileX
    rankTiles = np.arange(RANK, total_tiles, SIZE) # tiles for this rank

    for row in range(numCoords): # each multipt. Only one for large scan

        ptName = f'Point{str(row).zfill(3)}'
        if useCoordNames and pd.notna(coords['ID'].iloc[row]):
            ptName = coords.ID.iloc[row]

        # Iterate through all possible tile indices (0 to Y*X - 1)
        # for tile_index in range(total_tiles):
        for tile_index in tqdm(rankTiles): # subset
            # Assign tile_index to a rank using modulo arithmetic (cyclic distribution)
            # Rank 'r' processes tiles where tile_index % size == r
            # Calculate the (row, col) from the linear tile_index
            ii = tile_index // tileX
            jj = tile_index % tileX

            # This rank is responsible for this tile, so process it
            result = process_single_tile_large_scan(ptName, useCoordNames, ii, jj, tileX, 
                            dimZ, channelFilters, nRepFrames, rawFiles, 
                            localPath, rawPath, rawTifPath, 
                            processedTilePath, renderPath, 
                            volumeTilePath, coordIdx = row)

            # Store the result if needed
            local_results.append(result)

    # --- Gather Results (Optional) ---
    # If you need to collect results from all tiles onto one process (e.g., rank 0)
    # Use gather - it collects the list 'local_results' from each rank
    # into a list of lists on the root process.

    # print(f"Rank {rank}: Finished processing. Sending {len(local_results)} results.") # Debug print
    all_gathered_results = COMM.gather(local_results, root=0)
    # ---------------------------------

    # --- Final Processing on Root Rank (Optional) ---
    if RANK == 0:
        print(f"\n--- Rank 0: All processes finished. Gathering results. ---")
        final_results = []
        if all_gathered_results:
            # all_gathered_results is a list of lists, e.g., [[rank0_res1, rank0_res2], [rank1_res1], ...]
            # Flatten the list of lists into a single list
            for rank_results in all_gathered_results:
                final_results.extend(rank_results)

        print(f"Total results gathered: {len(final_results)}")

        # Optional: Sort results if order matters (e.g., by row, then col)
        final_results.sort(key=lambda item: (item[0], item[1]))

        # Now you can process 'final_results' which contains results from all tiles
        print(f"First 5 results (if any): {final_results[:5]}")
        # print("\nAll gathered results (sorted):")
        # for res in final_results:
        #     print(res)

        # Verification
        if len(final_results) == total_tiles:
            print(f"Verification successful: Received results for all {total_tiles} tiles.")
        else:
            print(f"Verification WARNING: Expected {total_tiles} results, but received {len(final_results)}.")

# %% stitch 3D volumes
# edit macro to resume where left off
def editStitchMacro(macroName, ptName):

    # read macro
    with open(macroName, 'r') as file:
        content = file.readlines()

    # overwrite previous macro
    for ii, line in enumerate(content):
        if ptName in line: # keep lines from here onward
            break
        else: # comment previous
            content[ii] = '//' + line if '//' not in line else line

    # write updated macro
    with open(macroName, 'w') as file:
        file.writelines(content)

# Stack all stitched Z planes to create final 3D volume
def stackVolumePlanes(folderPath, basePath, started, ptName, macroName, channelFilters, startTime, 
                      yTileDim, xTileDim, tileY, tileX, overlapFrac):

    # check if volumes have been stitched
    tifFiles = [f for f in folderPath.glob('*.tif')]
    if len(tifFiles) == len(channelFilters):
        # check all file sizes match
        sizes = [f.stat().st_size for f in tifFiles]
        if len(set(sizes)) == 1: # all files are the same size
            # comment previous macro lines up to this coord
            editStitchMacro(macroName, ptName)
            return # all files are ready. Coord has been stitched already

    # files may not be ready yet
    # get most recent modified file
    crashed = False
    while True:
        fileNames = [f for f in folderPath.glob('*') if f.suffix == '']
        fileNames = sorted(fileNames, key = lambda x: x.stat().st_mtime, reverse = True)
        if len(fileNames) == 0: # processing not started yet
            time.sleep(10)
            if not crashed and started and time.time() - startTime > 45 * 60: # Fiji started processing AND more than 45 mins has passed -> Fiji likely crashed
                    # sendEmailAlert(subject = 'Fiji crashed during 3D volume stitching',
                    #                startBody = f'Processing crashed during 3D volume stitching at {ptName}', 
                    #                basePath=basePath)
                    crashed = True
                    # edit the Fiji macro to start running at this point. Comment out previous lines
                    editStitchMacro(macroName, ptName)
            continue

        latest_file = max(fileNames, key=os.path.getctime)
        # check if file modification was more than 10 mins ago
        if time.time() - latest_file.stat().st_mtime > 10 * 60:
            break # no new files have been added. Assume all files are ready

    fileNames.sort() # T, Z, C

    dimZ = 0
    dimC = 0
    for jj, name in enumerate(fileNames):
        _, _, z, c = name.stem.split('_')
        z = z.replace('z', '')
        c = c.replace('c', '')

        if int(z) > dimZ:
            dimZ = int(z)
        if int(c) > dimC:
            dimC = int(c)

    assert dimC == len(channelFilters), 'Missing channels in stitched folder'
    # all files should be same file size
    sizes = [f.stat().st_size for f in fileNames]
    assert len(set(sizes)) == 1, 'Files are not the same size'

    # chStack = []
    for jj, excite in enumerate(channelFilters): # each channel
        # zStack = []
        # export each channel separately
        fileOut = folderPath / f'{ptName}_{excite}_stitched.tif'
        if fileOut.exists():
            os.remove(fileOut) # delete file
            
        pauseCheckRAM()
        for kk in range(dimZ): # each z plane

            fileName = [f for f in fileNames if f.name.endswith(f'z{str(kk + 1).zfill(2)}_c{str(jj + 1)}')]
            if len(fileName) == 0:
                fileName = [f for f in fileNames if f.name.endswith(f'z{str(kk + 1).zfill(3)}_c{str(jj + 1)}')]
            assert len(fileName) == 1
            plane = tf.imread(fileName[0]) # YX
            yLargeDim, xLargeDim = plane.shape

            # append each plane to file
            tf.imwrite(fileOut, plane, photometric = 'minisblack', append = True, bigtiff = True, metadata = None)

    # check that stitching dimensions are approx correct
    yTileApprox = np.round(yLargeDim / ((1 - overlapFrac) * yTileDim)).astype(int)
    xTileApprox = np.round(xLargeDim / ((1 - overlapFrac) * xTileDim)).astype(int)
    if yTileApprox != tileY or xTileApprox != tileX:
        print('Warning: stitching dimensions are estimated to be wrong for', ptName)

# skip on cluster
'''
if useCoordNames and tileY > 1 and tileX > 1 and rank == 0: # multipoint tiles
    
    print('Stacking all stitched Z planes to create final 3D volumes...')
    started = False
    with tqdm(total = len(coords)) as pbar:
        for ii, row1 in coords.iterrows(): # each coord

            ptName = f'Point{str(ii).zfill(3)}'
            if useCoordNames and pd.notna(row1.ID):
                ptName = row1.ID

            folderPath = volumeStitchPath / ptName
            folderPath.mkdir(exist_ok = True)

            startTime = time.time()
            stackVolumePlanes(folderPath, basePath, started, ptName, macroName, channelFilters, startTime, 
                                yTileDim, xTileDim, tileY, tileX, overlapFrac)
            started = True # Fiji started processing
            pbar.update(1)'
'''