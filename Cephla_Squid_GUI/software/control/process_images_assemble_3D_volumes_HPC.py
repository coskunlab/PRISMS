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
from joblib import Parallel, delayed

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

python3 process_images_real_time.py /home/cephla/Desktop/Nicky/48%20NFkB%20gradient%20on%20chip/Data/01-3T3%20P26%20coverslip%20112F/31Jan2025%20cycle%204%20HCR/_2025-01-24_15-05-01.656011/0 0.15 40 3 3 1 16 MIP /home/cephla/Downloads/01-3T3%20P26%20coverslip%20112F/31Jan2025%20cycle%204%20HCR/_2025-01-24_15-05-01.656011/0
'''
#%%
# --- MPI Initialization ---
COMM = MPI.COMM_WORLD # Default communicator
RANK = COMM.Get_rank() # Get the rank (ID) of the current process
SIZE = COMM.Get_size() # Get the total number of processes
# print('Rank', rank, 'of size', size, 'started.')
assert SIZE > 1, 'This script must be run with MPI (multiple processes).'

# %% [markdown]
# # user directories and inputs
# get arguments passed from command line
# %%
# in raw folder
script_dir = os.path.dirname(os.path.abspath(__file__))
basePath = Path(script_dir).parent
assert basePath.exists()

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

# base folder
# basePath = rawPath.parent.parent

# overlap fraction between frames
# overlapFrac = float(sys.argv[2])
overlapFrac = 0.15
# print('Overlap fraction:', overlapFrac)

# dimZ = int(sys.argv[3])
# tileY = int(sys.argv[4])
# tileX = int(sys.argv[5])
# nRepFrames = int(sys.argv[6])
# dimZ = 40
# tileY = 5
# tileX = 5
jsonFile = [f for f in rawPath.parent.glob('*.json')]
assert len(jsonFile) == 1
with open(jsonFile[0]) as f:
    meta = json.load(f)
    dimZ = meta['Nz']
    tileY = meta['Ny']
    tileX = meta['Nx']
nRepFrames = 1

# local save path
# localPath = sys.argv[9]
# localPath = urllib.parse.unquote(localPath)
localPath = r''
localPath = Path(localPath)
assert localPath.exists()

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
    # volumeTilePath.mkdir(exist_ok = True)
    assert volumeTilePath.exists(), 'Volume tile path does not exist. Please create it first.'

    # folder for 3D stitching volumes
    volumeStitchPath = basePath / '07 stitched 3D volumes Grid Collection'
    # volumeStitchPath.mkdir(exist_ok=True)
    assert volumeStitchPath.exists(), 'Volume stitch path does not exist. Please create it first.'

else: # large scan
    useCoordNames = False

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
    assert dimY == dimX
    if dimY == 2304:
        print('Hamamatsu camera detected')
        cameraType = 'Hamamatsu'
    elif dimY == 2084:
        print('Sony camera detected')
        cameraType = 'Sony'
    else:
        sys.exit('Unknown camera type')
        cameraType = 'Unknown'
    return cameraType, dimY, dimX

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

cameraType, yTileDim, xTileDim = getCameraTypeOrientation(rawPath)

# %% [markdown]
# # compute time b/t first z planes of diff tiles

#%% use tqdm
# detect if software froze
def detectSoftwareFreeze(start, end, row, ii, jj, basePath, subject, startBody):
    if end - start > 15 * 60: # longer than 15 minutes per tile. Software likely froze
        print('Tile', [str(row), str(ii), str(jj), '0'], 'taking too long. Software may have frozen. Sending email at', datetime.now())
        sendEmailAlert(subject = subject, startBody = startBody, basePath = basePath)
        return True
    return False

# send email alert to user
def sendEmailAlert(subject, startBody, basePath):
    # send email alert
    sender = 'nzhang326@gatech.edu'
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
            server.login(sender, "Outrank-Undoing-Satin") # password
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

# reformat Linux home paths to Windows
def reformatPathLinuxToWindows(linuxPath):

    # check OS
    if platform.system() == 'Linux':
        # Linux has different parent path than windows. Reformat parent to match windows so user can run Fiji in windows
        path = linuxPath.parts
        # find Desktop string match
        for idxCoskun, part in enumerate(path):
            if part == 'Desktop':
                break
        if 'coskun-lab' in path:
            baseFolder = 'coskun-lab'
        elif 'coskun-lab2' in path:
            baseFolder = 'coskun-lab2'
        else:
            raise ValueError('Unknown base folder')
        # add first half of path
        path = Path('Y:', baseFolder, *path[idxCoskun + 1:]).parts
        # remove duplicate entries
        ordered = []
        for part in path:
            if part not in ordered:
                ordered.append(part)
        windowsPath = Path(*ordered)

    elif platform.system() == 'Windows':
        windowsPath = linuxPath # no reformatting needed

    else:
        raise ValueError('Unknown OS')

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

    frozen = False # assume software not frozen
    while True:

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
        
        else: # found file
            time.sleep(1) # wait for file to be written
            break

        end  = time.time()
        if not frozen:
            frozen = detectSoftwareFreeze(start, end, row, ii, jj, basePath, 
                                          subject = "Detected Squid confocal software crash", 
                                          startBody = "Squid software crashed")

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
        splitFolder = str(splitFolder).replace('\\', '/')
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
        macro.write(str(folderOut).replace('\\', '/'))
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
    splitFolder = str(splitFolder).replace('\\', '/')
    macro.write(str(splitFolder))
    macro.write('] filenamepattern=tile_{ppppp}_CH')
    macro.write(str(jj + 1))
    macro.write('.tif filenamepatterntype=SEQUENTIAL gridorigin=')
    if cameraType == 'Hamamatsu':
        macro.write('LR')
    elif cameraType == 'Sony':
        macro.write('UR')
    macro.write(' assemblefrommetadata=true assemblenooverlap=false globalpositionsfile=[')
    macro.write(str(folderOut / f'{channelFilters[0]}_img_t1_z1_c1_global-positions-0.txt').replace('\\', '/'))
    macro.write('] numberingpattern=HORIZONTALCONTINUOUS startrow=0 startcol=0 extentwidth=')
    macro.write(str(tileX))
    macro.write(' extentheight=')
    macro.write(str(tileY))
    macro.write(' timeslices=0 istimeslicesenabled=false outputpath=[')
    macro.write(str(folderOut).replace('\\', '/'))
    macro.write('] displaystitching=false outputfullimage=true outputmeta=true outputimgpyramid=false blendingmode=LINEAR blendingalpha=1.5 compressionmode=UNCOMPRESSED outfileprefix=')
    macro.write(f'{excite}_')
    macro.write('img_t1_z')
    macro.write(str(jj + 1))
    macro.write('_c1_ unit=MICROMETER unitx=1.0 unity=1.0 programtype=AUTO numcputhreads=32 loadfftwplan=true savefftwplan=true fftwplantype=MEASURE fftwlibraryname=libfftw3 fftwlibraryfilename=libfftw3.dll planpath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw\\fftPlans fftwlibrarypath=C:\\Users\\nzhang326\\Fiji.app\\lib\\fftw stagerepeatability=0 horizontaloverlap=')
    macro.write(str(overlapFrac * 100))
    macro.write(' verticaloverlap=')
    macro.write(str(overlapFrac * 100))
    macro.write(' numfftpeaks=0 overlapuncertainty=NaN isusedoubleprecision=false isusebioformats=false issuppressmodelwarningdialog=true isenablecudaexceptions=false translationrefinementmethod=SINGLE_HILL_CLIMB numtranslationrefinementstartpoints=16 headless=false loglevel=MANDATORY debuglevel=NONE"); \n')

# stitch with Grid Collection
def stitch_3D_Grid_Collection(macro, tileX, tileY, overlapFrac, cameraType, tilePath, stitchPath, ptName, rgbMode = False):

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
    splitFolder = str(splitFolder).replace('\\', '/')
    macro.write(' directory=[' + splitFolder + '] file_names=tile_{iiiii}.tif')
    macro.write(' output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50')
    # macro.write(' compute_overlap') # sometimes messes up stitching
    macro.write(' computation_parameters=[Save memory (but be slower)] image_output=[Write to disk]')
    volOutPath = reformatPathLinuxToWindows(Path(stitchPath, ptName))
    if platform.system() == 'Windows': volOutPath.mkdir(exist_ok = True)
    macro.write(' output_directory=[' + str(volOutPath).replace('\\', '/') + ']"); \n')

'''
# write Fiji macro
now = datetime.now() # current date and time
date_time = now.strftime("%d%b%Y")
macroName = Path(largeImagePath, date_time + '_stitch.ijm')
macro = open(macroName, 'w')

# add run macro line as comment in first line of actual macro
runCommand = reformatPathLinuxToWindows(linuxPath=largeImagePath)
# runCommand = largeImagePath
runCommand = runCommand / macroName.name
macro.write('// runMacro("' + str(runCommand).replace('\\', '/') + '"); \n\n')

for row in range(numCoords): # each multipt. Only one for large scan

    ptName = f'Point{str(row).zfill(3)}'
    if useCoordNames and pd.notna(coords['ID'].iloc[row]):
        ptName = coords.ID.iloc[row]

    if useCoordNames:
        folderOut = largeImagePath / str(coords['ID'].iloc[row] if pd.notna(coords['ID'].iloc[row]) else ptName)
        # create 3D folder
        Path(volumeStitchPath / folderOut.stem).mkdir(exist_ok = True)
    else:
        folderOut = largeImagePath / ptName
    folderOut.mkdir(exist_ok = True)    

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
        stitch_3D_Grid_Collection(macro, tileX, tileY, overlapFrac, cameraType, volumeTilePath, volumeStitchPath, folderOut.stem)

# quit Fiji when finished to save RAM
# macro.write('run("Quit"); \n')
macro.close()

# print out command to run this macro in Fiji
print('runMacro("' + str(runCommand).replace('\\', '/') + '");')'
'''

#%% Processing mode
'''
totalIter = numCoords * tileY * tileX

lazy_read = dask.delayed(tf.imread)

# rawFiles = [f for f in rawPath.glob('*.tiff')]
rawFiles = getRawFiles(localPath, rawPath)
networkFiles = [f for f in rawPath.glob('*.tiff')]
print('Sample', str(basePath))

#### give user option to continue processing or only move files
onlyMoveFiles, timedOut = timedInput('Only moves files (m) or process images on this computer (p)?:', timeout = 5 * 60)
if timedOut:
    if platform.system() == 'Linux':
        onlyMoveFiles = 'm' # default only move files on Linux
        print('Timed out. Only moving files...')
    elif platform.system() == 'Windows':
        onlyMoveFiles = 'p' # default process files on Windows
        print('Timed out. Processing all files...')
    else: 
        raise ValueError('Unknown OS')

#%% simultaneous processing w/ acquisition
with tqdm(total=totalIter) as pbar:

    for row in range(numCoords): # each multipoint

        ptName = f'Point{str(row).zfill(3)}'
        if useCoordNames and pd.notna(coords['ID'].iloc[row]):
            ptName = coords.ID.iloc[row]

        xScanDir = 1 # snake by rows
        fileNum = 0 # each frame step
        for ii in range(tileY): # each Y tile

            for jj in range(tileX): # each X tile

                # print disk usage
                print('Sample', str(basePath))
                try:
                    print('Disk usage:', psutil.disk_usage('/').percent, '%')
                except:
                    pass
                # print RAM usage
                print('RAM usage:', psutil.virtual_memory().percent, '%')

                # temp pause if RAM usage is too high
                pauseCheckRAM()

                # time each tile. If takes too long then send email to user
                start = time.time()
                
                # compute x tile index based on scan direction
                jj = jj if xScanDir == 1 else tileX - 1 - jj

                # locate tile files
                while True:
                    try:
                        rawFiles = locateTileFiles(rawFiles, row, ii, jj, dimZ, channelFilters, start, basePath, nRepFrames)
                    except Exception as e:
                        print('Error:', e)
                        print('Retry locate tile files...')
                        time.sleep(10 * 60)
                        continue
                    else: # success
                        break
        
                # print('Found', [str(row), str(ii), str(jj), '0'])
                ##### move files to network
                pbar.update(1)
                while True:
                    try:
                        copyLocalFilesToNetwork(localPath, rawPath, ii, jj, row)
                    except Exception as e:
                        print('Error:', e)
                        print('Retry moving files...')
                        time.sleep(10 * 60)
                        continue
                    else: # success
                        sys.stdout.flush()      
                        break

                # only move files or do all processing?
                if onlyMoveFiles == 'm':
                    continue
                
                #### export raw TIFF z stack
                while True:
                    try:
                        chStack = combineZStack(row, ii, jj, dimZ, channelFilters, 
                                                networkFiles, rawPath, rawTifPath, ptName, 
                                                useCoordNames, nRepFrames) # CZYX unless RGB
                    except Exception as e:
                        print('Error:', e)
                        print('Retry combine Z stack...')
                        time.sleep(10 * 60)
                        continue
                    else: # success
                        break
                
                ##### merge RGB if applicable
                if rgbMode: 
                    while True:
                        try:
                            chStack = mergeRGB(rawTifPath, ptName, ii, jj) # YX3
                        except Exception as e:
                            print('Error:', e)
                            print('Retry merge RGB...')
                            time.sleep(10 * 60)
                            continue
                        else: # success
                            break

                #### compute tiles for stitching
                fileNum += 1
                while True:
                    try:
                        reformatTiles(stitchMode, chStack, row, ii, jj, fileNum, channelFilters, rawTilePath, 
                                      processedTilePath, renderPath, ptName, useCoordNames, 
                                      volumeTilePath = volumeTilePath if useCoordNames else None, 
                                      rgbMode = rgbMode)
                    except Exception as e:
                        print('Error:', e)
                        print('Retry reformat tiles...')
                        time.sleep(10 * 60)
                        continue
                    else: # success
                        break
                
                # record tile dimensions for later
                _, _, yTileDim, xTileDim = chStack.shape # CZYX
                del chStack # save RAM

            xScanDir *= -1 # reverse scan direction
'''
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
def stackVolumePlanes(folderPath, basePath, ptName, channelFilters, startTime, 
                      yTileDim, xTileDim, tileY, tileX, overlapFrac):

    # # check if volumes have been stitched
    # tifFiles = [f for f in folderPath.glob('*.tif')]
    # if len(tifFiles) == len(channelFilters):
    #     # check all file sizes match
    #     sizes = [f.stat().st_size for f in tifFiles]
    #     if len(set(sizes)) == 1: # all files are the same size
    #         # comment previous macro lines up to this coord
    #         # editStitchMacro(macroName, ptName)
    #         return # all files are ready. Coord has been stitched already

    fileNames = [f for f in folderPath.glob('*') if f.suffix == '']
    # fileNames = sorted(fileNames, key = lambda x: x.stat().st_mtime, reverse = True)
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
            
        # pauseCheckRAM()
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

def stack_volumes_per_coord(RANK, basePath, volumeStitchPath, row1, channelFilters,
                            yTileDim, xTileDim, tileY, tileX, overlapFrac,
                            useCoordNames):
    print(f"Rank {RANK}: Processing coord {row1.Index}...")
    ptName = f'Point{str(row1.Index).zfill(3)}'
    if useCoordNames and pd.notna(row1.ID):
        ptName = row1.ID

    folderPath = volumeStitchPath / ptName
    folderPath.mkdir(exist_ok = True)

    startTime = time.time()
    res = stackVolumePlanes(folderPath, basePath, ptName, channelFilters, startTime, 
                        yTileDim, xTileDim, tileY, tileX, overlapFrac)
    local_results.append(res)

if RANK == 0:
  print(f"Starting parallel computation with {SIZE} processes...")
  print(f"Total coords rows: {len(coords)}")

# Add a barrier for cleaner print output (optional)
COMM.Barrier()
rankCoords = np.arange(RANK, len(coords), SIZE) 
rankCoords = coords.iloc[rankCoords]
print(f"Rank {RANK}: Processing coords:")
print(rankCoords)
print()

local_results = []
if useCoordNames and tileY > 1 and tileX > 1: # multipoint tiles
    
    print('Stacking all stitched Z planes to create final 3D volumes...')
    # for row1 in coords.itertuples(index=True): # each coord
    for row1 in rankCoords.itertuples(index=True): # subset
        res = stack_volumes_per_coord(RANK, basePath, volumeStitchPath, row1, channelFilters,
                            yTileDim, xTileDim, tileY, tileX, overlapFrac,
                            useCoordNames)
        local_results.append(res)
    # Parallel(n_jobs = -1, prefer = 'threads', verbose = 50)\
    #      (delayed(stack_volumes_per_coord)(RANK, basePath, volumeStitchPath, row1, channelFilters,
    #                         yTileDim, xTileDim, tileY, tileX, overlapFrac,
    #                         useCoordNames) for row1 in coords.itertuples(index = True)) 

all_gathered_results = COMM.gather(local_results, root=0)

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
    # # Optional: Sort results if order matters (e.g., by row, then col)
    # final_results.sort(key=lambda item: (item[0], item[1]))

    # Verification
    if len(final_results) == len(coords):
         print(f"Verification successful: Received results for all {len(coords)} tiles.")
    else:
         print(f"Verification WARNING: Expected {len(coords)} results, but received {len(final_results)}.")
