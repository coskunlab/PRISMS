# set QT_API environment variable
import os
import sys

from control.microcontroller import Microcontroller
from squid.abc import AbstractStage
import squid.logging

# qt libraries
os.environ["QT_API"] = "pyqt5"
import qtpy
import pyqtgraph as pg
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

# control
from control._def import *

import asyncio
import urllib.parse
import shutil
import pip
from tqdm import tqdm, trange
import dask
import dask.array as da
import tifffile as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import platform
try:
    __import__('pytimedinput')
except ImportError:
    pip.main(['install', 'pytimedinput'])
from pytimedinput import timedInput

DO_FLUORESCENCE_RTP = True # does not work from _def.py
if DO_FLUORESCENCE_RTP:
    from control.processing_handler import ProcessingHandler
    # from control.processing_pipeline import *
    # from control.multipoint_built_in_functionalities import malaria_rtp

import control.utils as utils
import control.utils_config as utils_config
import control.tracking as tracking
import control.serial_peripherals as serial_peripherals

try:
    from control.multipoint_custom_script_entry_v2 import *

    print("custom multipoint script found")
except:
    pass

from typing import List, Tuple, Optional
from queue import Queue
from threading import Thread, Lock
from pathlib import Path
from datetime import datetime
import time
import subprocess
import shutil
import itertools
from lxml import etree
import json
import math
import random
import numpy as np
import pandas as pd
import scipy.signal
import cv2
import imageio as iio
import squid.abc

class ObjectiveStore:
    def __init__(self, objectives_dict=OBJECTIVES, default_objective=DEFAULT_OBJECTIVE, parent=None):
        self.objectives_dict = objectives_dict
        self.default_objective = default_objective
        self.current_objective = default_objective
        self.tube_lens_mm = TUBE_LENS_MM
        self.sensor_pixel_size_um = CAMERA_PIXEL_SIZE_UM[CAMERA_SENSOR]
        self.pixel_binning = self.get_pixel_binning()
        self.pixel_size_um = self.calculate_pixel_size(self.current_objective)

    def get_pixel_size(self):
        return self.pixel_size_um

    def calculate_pixel_size(self, objective_name):
        objective = self.objectives_dict[objective_name]
        magnification = objective["magnification"]
        objective_tube_lens_mm = objective["tube_lens_f_mm"]
        pixel_size_um = self.sensor_pixel_size_um / (magnification / (objective_tube_lens_mm / self.tube_lens_mm))
        pixel_size_um *= self.pixel_binning
        return pixel_size_um

    def set_current_objective(self, objective_name):
        if objective_name in self.objectives_dict:
            self.current_objective = objective_name
            self.pixel_size_um = self.calculate_pixel_size(objective_name)
        else:
            raise ValueError(f"Objective {objective_name} not found in the store.")

    def get_current_objective_info(self):
        return self.objectives_dict[self.current_objective]

    def get_pixel_binning(self):
        try:
            highest_res = max(self.parent.camera.res_list, key=lambda res: res[0] * res[1])
            resolution = self.parent.camera.resolution
            pixel_binning = max(1, highest_res[0] / resolution[0])
        except AttributeError:
            pixel_binning = 1
        return pixel_binning


class StreamHandler(QObject):

    image_to_display = Signal(np.ndarray)
    packet_image_to_write = Signal(np.ndarray, int, float)
    packet_image_for_tracking = Signal(np.ndarray, int, float)
    signal_new_frame_received = Signal()

    def __init__(
        self, crop_width=Acquisition.CROP_WIDTH, crop_height=Acquisition.CROP_HEIGHT, display_resolution_scaling=1
    ):
        QObject.__init__(self)
        self.fps_display = 1
        self.fps_save = 1
        self.fps_track = 1
        self.timestamp_last_display = 0
        self.timestamp_last_save = 0
        self.timestamp_last_track = 0

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.display_resolution_scaling = display_resolution_scaling

        self.save_image_flag = False
        self.track_flag = False
        self.handler_busy = False

        # for fps measurement
        self.timestamp_last = 0
        self.counter = 0
        self.fps_real = 0

    def start_recording(self):
        self.save_image_flag = True

    def stop_recording(self):
        self.save_image_flag = False

    def start_tracking(self):
        self.tracking_flag = True

    def stop_tracking(self):
        self.tracking_flag = False

    def set_display_fps(self, fps):
        self.fps_display = fps

    def set_save_fps(self, fps):
        self.fps_save = fps

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling / 100
        print(self.display_resolution_scaling)

    def on_new_frame(self, camera):

        if camera.is_live:

            camera.image_locked = True
            self.handler_busy = True
            self.signal_new_frame_received.emit()  # self.liveController.turn_off_illumination()

            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter + 1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                if PRINT_CAMERA_FPS:
                    # print("real camera fps is " + str(self.fps_real))
                    pass

            # moved down (so that it does not modify the camera.current_frame, which causes minor problems for simulation) - 1/30/2022
            # # rotate and flip - eventually these should be done in the camera
            # camera.current_frame = utils.rotate_and_flip_image(camera.current_frame,rotate_image_angle=camera.rotate_image_angle,flip_image=camera.flip_image)

            # crop image
            image_cropped = utils.crop_image(camera.current_frame, self.crop_width, self.crop_height)
            image_cropped = np.squeeze(image_cropped)

            # # rotate and flip - moved up (1/10/2022)
            # image_cropped = utils.rotate_and_flip_image(image_cropped,rotate_image_angle=ROTATE_IMAGE_ANGLE,flip_image=FLIP_IMAGE)
            # added on 1/30/2022
            # @@@ to move to camera
            image_cropped = utils.rotate_and_flip_image(
                image_cropped, rotate_image_angle=camera.rotate_image_angle, flip_image=camera.flip_image
            )

            # send image to display
            time_now = time.time()
            if time_now - self.timestamp_last_display >= 1 / self.fps_display:
                # self.image_to_display.emit(cv2.resize(image_cropped,(round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling)),cv2.INTER_LINEAR))
                self.image_to_display.emit(
                    utils.crop_image(
                        image_cropped,
                        round(self.crop_width * self.display_resolution_scaling),
                        round(self.crop_height * self.display_resolution_scaling),
                    )
                )
                self.timestamp_last_display = time_now

            # send image to write
            if self.save_image_flag and time_now - self.timestamp_last_save >= 1 / self.fps_save:
                if camera.is_color:
                    image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
                self.packet_image_to_write.emit(image_cropped, camera.frame_ID, camera.timestamp)
                self.timestamp_last_save = time_now

            # send image to track
            if self.track_flag and time_now - self.timestamp_last_track >= 1 / self.fps_track:
                # track is a blocking operation - it needs to be
                # @@@ will cropping before emitting the signal lead to speedup?
                self.packet_image_for_tracking.emit(image_cropped, camera.frame_ID, camera.timestamp)
                self.timestamp_last_track = time_now

            self.handler_busy = False
            camera.image_locked = False

    """
    def on_new_frame_from_simulation(self,image,frame_ID,timestamp):
        # check whether image is a local copy or pointer, if a pointer, needs to prevent the image being modified while this function is being executed

        self.handler_busy = True

        # crop image
        image_cropped = utils.crop_image(image,self.crop_width,self.crop_height)

        # send image to display
        time_now = time.time()
        if time_now-self.timestamp_last_display >= 1/self.fps_display:
            self.image_to_display.emit(cv2.resize(image_cropped,(round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling)),cv2.INTER_LINEAR))
            self.timestamp_last_display = time_now

        # send image to write
        if self.save_image_flag and time_now-self.timestamp_last_save >= 1/self.fps_save:
            self.packet_image_to_write.emit(image_cropped,frame_ID,timestamp)
            self.timestamp_last_save = time_now

        # send image to track
        if time_now-self.timestamp_last_display >= 1/self.fps_track:
            # track emit
            self.timestamp_last_track = time_now

        self.handler_busy = False
    """


class ImageSaver(QObject):

    stop_recording = Signal()

    def __init__(self, image_format=Acquisition.IMAGE_FORMAT):
        QObject.__init__(self)
        self.base_path = "./"
        self.experiment_ID = ""
        self.image_format = image_format
        self.max_num_image_per_folder = 1000
        self.queue = Queue(10)  # max 10 items in the queue
        self.image_lock = Lock()
        self.stop_signal_received = False
        self.thread = Thread(target=self.process_queue)
        self.thread.start()
        self.counter = 0
        self.recording_start_time = 0
        self.recording_time_limit = -1

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_ID, timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                folder_ID = int(self.counter / self.max_num_image_per_folder)
                file_ID = int(self.counter % self.max_num_image_per_folder)
                # create a new folder
                if file_ID == 0:
                    utils.ensure_directory_exists(os.path.join(self.base_path, self.experiment_ID, str(folder_ID)))

                if image.dtype == np.uint16:
                    # need to use tiff when saving 16 bit images
                    saving_path = os.path.join(
                        self.base_path, self.experiment_ID, str(folder_ID), str(file_ID) + "_" + str(frame_ID) + ".tiff"
                    )
                    iio.imwrite(saving_path, image)
                else:
                    saving_path = os.path.join(
                        self.base_path,
                        self.experiment_ID,
                        str(folder_ID),
                        str(file_ID) + "_" + str(frame_ID) + "." + self.image_format,
                    )
                    cv2.imwrite(saving_path, image)

                self.counter = self.counter + 1
                self.queue.task_done()
                self.image_lock.release()
            except:
                pass

    def enqueue(self, image, frame_ID, timestamp):
        try:
            self.queue.put_nowait([image, frame_ID, timestamp])
            if (self.recording_time_limit > 0) and (
                time.time() - self.recording_start_time >= self.recording_time_limit
            ):
                self.stop_recording.emit()
            # when using self.queue.put(str_), program can be slowed down despite multithreading because of the block and the GIL
        except:
            print("imageSaver queue is full, image discarded")

    def set_base_path(self, path):
        self.base_path = path

    def set_recording_time_limit(self, time_limit):
        self.recording_time_limit = time_limit

    def start_new_experiment(self, experiment_ID, add_timestamp=True):
        if add_timestamp:
            # generate unique experiment ID
            self.experiment_ID = experiment_ID + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        else:
            self.experiment_ID = experiment_ID
        self.recording_start_time = time.time()
        # create a new folder
        try:
            utils.ensure_directory_exists(os.path.join(self.base_path, self.experiment_ID))
            # to do: save configuration
        except:
            pass
        # reset the counter
        self.counter = 0

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


class ImageSaver_Tracking(QObject):
    def __init__(self, base_path, image_format="bmp"):
        QObject.__init__(self)
        self.base_path = base_path
        self.image_format = image_format
        self.max_num_image_per_folder = 1000
        self.queue = Queue(100)  # max 100 items in the queue
        self.image_lock = Lock()
        self.stop_signal_received = False
        self.thread = Thread(target=self.process_queue)
        self.thread.start()

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_counter, postfix] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                folder_ID = int(frame_counter / self.max_num_image_per_folder)
                file_ID = int(frame_counter % self.max_num_image_per_folder)
                # create a new folder
                if file_ID == 0:
                    utils.ensure_directory_exists(os.path.join(self.base_path, str(folder_ID)))
                if image.dtype == np.uint16:
                    saving_path = os.path.join(
                        self.base_path,
                        str(folder_ID),
                        str(file_ID) + "_" + str(frame_counter) + "_" + postfix + ".tiff",
                    )
                    iio.imwrite(saving_path, image)
                else:
                    saving_path = os.path.join(
                        self.base_path,
                        str(folder_ID),
                        str(file_ID) + "_" + str(frame_counter) + "_" + postfix + "." + self.image_format,
                    )
                    cv2.imwrite(saving_path, image)
                self.queue.task_done()
                self.image_lock.release()
            except:
                pass

    def enqueue(self, image, frame_counter, postfix):
        try:
            self.queue.put_nowait([image, frame_counter, postfix])
        except:
            print("imageSaver queue is full, image discarded")

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


class ImageDisplay(QObject):

    image_to_display = Signal(np.ndarray)

    def __init__(self):
        QObject.__init__(self)
        self.queue = Queue(10)  # max 10 items in the queue
        self.image_lock = Lock()
        self.stop_signal_received = False
        self.thread = Thread(target=self.process_queue)
        self.thread.start()

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image, frame_ID, timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                self.image_to_display.emit(image)
                self.image_lock.release()
                self.queue.task_done()
            except:
                pass

    # def enqueue(self,image,frame_ID,timestamp):
    def enqueue(self, image):
        try:
            self.queue.put_nowait([image, None, None])
            # when using self.queue.put(str_) instead of try + nowait, program can be slowed down despite multithreading because of the block and the GIL
            pass
        except:
            print("imageDisplay queue is full, image discarded")

    def emit_directly(self, image):
        self.image_to_display.emit(image)

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()


class Configuration:
    def __init__(
        self,
        mode_id=None,
        name=None,
        color=None,
        camera_sn=None,
        exposure_time=None,
        analog_gain=None,
        illumination_source=None,
        illumination_intensity=None,
        z_offset=None,
        pixel_format=None,
        _pixel_format_options=None,
        emission_filter_position=None,
    ):
        self.id = mode_id
        self.name = name
        self.color = color
        self.exposure_time = exposure_time
        self.analog_gain = analog_gain
        self.illumination_source = illumination_source
        self.illumination_intensity = illumination_intensity
        self.camera_sn = camera_sn
        self.z_offset = z_offset
        self.pixel_format = pixel_format
        if self.pixel_format is None:
            self.pixel_format = "default"
        self._pixel_format_options = _pixel_format_options
        if _pixel_format_options is None:
            self._pixel_format_options = self.pixel_format
        self.emission_filter_position = emission_filter_position


class LiveController(QObject):
    def __init__(
        self,
        camera,
        microcontroller,
        configurationManager,
        illuminationController,
        parent=None,
        control_illumination=True,
        use_internal_timer_for_hardware_trigger=True,
        for_displacement_measurement=False,
    ):
        QObject.__init__(self)
        self.microscope = parent
        self.camera = camera
        self.microcontroller = microcontroller
        self.configurationManager = configurationManager
        self.currentConfiguration = None
        self.trigger_mode = TriggerMode.SOFTWARE  # @@@ change to None
        self.is_live = False
        self.control_illumination = control_illumination
        self.illumination_on = False
        self.illuminationController = illuminationController
        self.use_internal_timer_for_hardware_trigger = (
            use_internal_timer_for_hardware_trigger  # use QTimer vs timer in the MCU
        )
        self.for_displacement_measurement = for_displacement_measurement

        self.fps_trigger = 1
        self.timer_trigger_interval = (1 / self.fps_trigger) * 1000

        self.timer_trigger = QTimer()
        self.timer_trigger.setInterval(int(self.timer_trigger_interval))
        self.timer_trigger.timeout.connect(self.trigger_acquisition)

        self.trigger_ID = -1

        self.fps_real = 0
        self.counter = 0
        self.timestamp_last = 0

        self.display_resolution_scaling = DEFAULT_DISPLAY_CROP / 100

        self.enable_channel_auto_filter_switching = True

        if SUPPORT_SCIMICROSCOPY_LED_ARRAY:
            # to do: add error handling
            self.led_array = serial_peripherals.SciMicroscopyLEDArray(
                SCIMICROSCOPY_LED_ARRAY_SN, SCIMICROSCOPY_LED_ARRAY_DISTANCE, SCIMICROSCOPY_LED_ARRAY_TURN_ON_DELAY
            )
            self.led_array.set_NA(SCIMICROSCOPY_LED_ARRAY_DEFAULT_NA)

    # illumination control
    def turn_on_illumination(self):
        if self.illuminationController is not None and not "LED matrix" in self.currentConfiguration.name:
            self.illuminationController.turn_on_illumination(
                int(self.configurationManager.extract_wavelength(self.currentConfiguration.name))
            )
        elif SUPPORT_SCIMICROSCOPY_LED_ARRAY and "LED matrix" in self.currentConfiguration.name:
            self.led_array.turn_on_illumination()
        else:
            self.microcontroller.turn_on_illumination()
        self.illumination_on = True

    def turn_off_illumination(self):
        if self.illuminationController is not None and not "LED matrix" in self.currentConfiguration.name:
            self.illuminationController.turn_off_illumination(
                int(self.configurationManager.extract_wavelength(self.currentConfiguration.name))
            )
        elif SUPPORT_SCIMICROSCOPY_LED_ARRAY and "LED matrix" in self.currentConfiguration.name:
            self.led_array.turn_off_illumination()
        else:
            self.microcontroller.turn_off_illumination()
        self.illumination_on = False

    def set_illumination(self, illumination_source, intensity, update_channel_settings=True):
        if illumination_source < 10:  # LED matrix
            if SUPPORT_SCIMICROSCOPY_LED_ARRAY:
                # set color
                if "BF LED matrix full_R" in self.currentConfiguration.name:
                    self.led_array.set_color((1, 0, 0))
                elif "BF LED matrix full_G" in self.currentConfiguration.name:
                    self.led_array.set_color((0, 1, 0))
                elif "BF LED matrix full_B" in self.currentConfiguration.name:
                    self.led_array.set_color((0, 0, 1))
                else:
                    self.led_array.set_color(SCIMICROSCOPY_LED_ARRAY_DEFAULT_COLOR)
                # set intensity
                self.led_array.set_brightness(intensity)
                # set mode
                if "BF LED matrix left half" in self.currentConfiguration.name:
                    self.led_array.set_illumination("dpc.l")
                if "BF LED matrix right half" in self.currentConfiguration.name:
                    self.led_array.set_illumination("dpc.r")
                if "BF LED matrix top half" in self.currentConfiguration.name:
                    self.led_array.set_illumination("dpc.t")
                if "BF LED matrix bottom half" in self.currentConfiguration.name:
                    self.led_array.set_illumination("dpc.b")
                if "BF LED matrix full" in self.currentConfiguration.name:
                    self.led_array.set_illumination("bf")
                if "DF LED matrix" in self.currentConfiguration.name:
                    self.led_array.set_illumination("df")
            else:
                if "BF LED matrix full_R" in self.currentConfiguration.name:
                    self.microcontroller.set_illumination_led_matrix(illumination_source, r=(intensity / 100), g=0, b=0)
                elif "BF LED matrix full_G" in self.currentConfiguration.name:
                    self.microcontroller.set_illumination_led_matrix(illumination_source, r=0, g=(intensity / 100), b=0)
                elif "BF LED matrix full_B" in self.currentConfiguration.name:
                    self.microcontroller.set_illumination_led_matrix(illumination_source, r=0, g=0, b=(intensity / 100))
                else:
                    self.microcontroller.set_illumination_led_matrix(
                        illumination_source,
                        r=(intensity / 100) * LED_MATRIX_R_FACTOR,
                        g=(intensity / 100) * LED_MATRIX_G_FACTOR,
                        b=(intensity / 100) * LED_MATRIX_B_FACTOR,
                    )
        else:
            # update illumination
            if self.illuminationController is not None:
                self.illuminationController.set_intensity(
                    int(self.configurationManager.extract_wavelength(self.currentConfiguration.name)), intensity
                )
            elif ENABLE_NL5 and NL5_USE_DOUT and "Fluorescence" in self.currentConfiguration.name:
                wavelength = int(self.currentConfiguration.name[13:16])
                self.microscope.nl5.set_active_channel(NL5_WAVENLENGTH_MAP[wavelength])
                if NL5_USE_AOUT and update_channel_settings:
                    self.microscope.nl5.set_laser_power(NL5_WAVENLENGTH_MAP[wavelength], int(intensity))
                if ENABLE_CELLX:
                    self.microscope.cellx.set_laser_power(NL5_WAVENLENGTH_MAP[wavelength], int(intensity))
            else:
                self.microcontroller.set_illumination(illumination_source, intensity)

        # set emission filter position
        if ENABLE_SPINNING_DISK_CONFOCAL:
            try:
                self.microscope.xlight.set_emission_filter(
                    XLIGHT_EMISSION_FILTER_MAPPING[illumination_source],
                    extraction=False,
                    validate=XLIGHT_VALIDATE_WHEEL_POS,
                )
            except Exception as e:
                print("not setting emission filter position due to " + str(e))

        if USE_ZABER_EMISSION_FILTER_WHEEL and self.enable_channel_auto_filter_switching:
            try:
                if (
                    self.currentConfiguration.emission_filter_position
                    != self.microscope.emission_filter_wheel.current_index
                ):
                    if ZABER_EMISSION_FILTER_WHEEL_BLOCKING_CALL:
                        self.microscope.emission_filter_wheel.set_emission_filter(
                            self.currentConfiguration.emission_filter_position, blocking=True
                        )
                    else:
                        self.microscope.emission_filter_wheel.set_emission_filter(
                            self.currentConfiguration.emission_filter_position, blocking=False
                        )
                        if self.trigger_mode == TriggerMode.SOFTWARE:
                            time.sleep(ZABER_EMISSION_FILTER_WHEEL_DELAY_MS / 1000)
                        else:
                            time.sleep(
                                max(0, ZABER_EMISSION_FILTER_WHEEL_DELAY_MS / 1000 - self.camera.strobe_delay_us / 1e6)
                            )
            except Exception as e:
                print("not setting emission filter position due to " + str(e))

        if (
            USE_OPTOSPIN_EMISSION_FILTER_WHEEL
            and self.enable_channel_auto_filter_switching
            and OPTOSPIN_EMISSION_FILTER_WHEEL_TTL_TRIGGER == False
        ):
            try:
                if (
                    self.currentConfiguration.emission_filter_position
                    != self.microscope.emission_filter_wheel.current_index
                ):
                    self.microscope.emission_filter_wheel.set_emission_filter(
                        self.currentConfiguration.emission_filter_position
                    )
                    if self.trigger_mode == TriggerMode.SOFTWARE:
                        time.sleep(OPTOSPIN_EMISSION_FILTER_WHEEL_DELAY_MS / 1000)
                    elif self.trigger_mode == TriggerMode.HARDWARE:
                        time.sleep(
                            max(0, OPTOSPIN_EMISSION_FILTER_WHEEL_DELAY_MS / 1000 - self.camera.strobe_delay_us / 1e6)
                        )
            except Exception as e:
                print("not setting emission filter position due to " + str(e))

        if USE_SQUID_FILTERWHEEL and self.enable_channel_auto_filter_switching:
            try:
                self.microscope.squid_filter_wheel.set_emission(self.currentConfiguration.emission_filter_position)
            except Exception as e:
                print("not setting emission filter position due to " + str(e))

    def start_live(self):
        self.is_live = True
        self.camera.is_live = True
        self.camera.start_streaming()
        if self.trigger_mode == TriggerMode.SOFTWARE or (
            self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger
        ):
            self.camera.enable_callback()  # in case it's disabled e.g. by the laser AF controller
            self._start_triggerred_acquisition()
        # if controlling the laser displacement measurement camera
        if self.for_displacement_measurement:
            self.microcontroller.set_pin_level(MCU_PINS.AF_LASER, 1)

    def stop_live(self):
        if self.is_live:
            self.is_live = False
            self.camera.is_live = False
            if hasattr(self.camera, "stop_exposure"):
                self.camera.stop_exposure()
            if self.trigger_mode == TriggerMode.SOFTWARE:
                self._stop_triggerred_acquisition()
            # self.camera.stop_streaming() # 20210113 this line seems to cause problems when using af with multipoint
            if self.trigger_mode == TriggerMode.CONTINUOUS:
                self.camera.stop_streaming()
            if (self.trigger_mode == TriggerMode.SOFTWARE) or (
                self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            if self.control_illumination:
                self.turn_off_illumination()
            # if controlling the laser displacement measurement camera
            if self.for_displacement_measurement:
                self.microcontroller.set_pin_level(MCU_PINS.AF_LASER, 0)

    # software trigger related
    def trigger_acquisition(self):
        if self.trigger_mode == TriggerMode.SOFTWARE:
            if self.control_illumination and self.illumination_on == False:
                self.turn_on_illumination()
            self.trigger_ID = self.trigger_ID + 1
            self.camera.send_trigger()
            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter + 1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                # print('real trigger fps is ' + str(self.fps_real))
        elif self.trigger_mode == TriggerMode.HARDWARE:
            self.trigger_ID = self.trigger_ID + 1
            if ENABLE_NL5 and NL5_USE_DOUT:
                self.microscope.nl5.start_acquisition()
            else:
                self.microcontroller.send_hardware_trigger(
                    control_illumination=True, illumination_on_time_us=self.camera.exposure_time * 1000
                )

    def _start_triggerred_acquisition(self):
        self.timer_trigger.start()

    def _set_trigger_fps(self, fps_trigger):
        self.fps_trigger = fps_trigger
        self.timer_trigger_interval = (1 / self.fps_trigger) * 1000
        self.timer_trigger.setInterval(int(self.timer_trigger_interval))

    def _stop_triggerred_acquisition(self):
        self.timer_trigger.stop()

    # trigger mode and settings
    def set_trigger_mode(self, mode):
        if mode == TriggerMode.SOFTWARE:
            if self.is_live and (
                self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            self.camera.set_software_triggered_acquisition()
            if self.is_live:
                self._start_triggerred_acquisition()
        if mode == TriggerMode.HARDWARE:
            if self.trigger_mode == TriggerMode.SOFTWARE and self.is_live:
                self._stop_triggerred_acquisition()
            # self.camera.reset_camera_acquisition_counter()
            self.camera.set_hardware_triggered_acquisition()
            self.reset_strobe_arugment()
            self.camera.set_exposure_time(self.currentConfiguration.exposure_time)

            if self.is_live and self.use_internal_timer_for_hardware_trigger:
                self._start_triggerred_acquisition()
        if mode == TriggerMode.CONTINUOUS:
            if (self.trigger_mode == TriggerMode.SOFTWARE) or (
                self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger
            ):
                self._stop_triggerred_acquisition()
            self.camera.set_continuous_acquisition()
        self.trigger_mode = mode

    def set_trigger_fps(self, fps):
        if (self.trigger_mode == TriggerMode.SOFTWARE) or (
            self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger
        ):
            self._set_trigger_fps(fps)

    # set microscope mode
    # @@@ to do: change softwareTriggerGenerator to TriggerGeneratror
    def set_microscope_mode(self, configuration):

        self.currentConfiguration = configuration
        # print("setting microscope mode to " + self.currentConfiguration.name)

        # temporarily stop live while changing mode
        if self.is_live is True:
            self.timer_trigger.stop()
            if self.control_illumination:
                self.turn_off_illumination()

        # set camera exposure time and analog gain
        self.camera.set_exposure_time(self.currentConfiguration.exposure_time)
        self.camera.set_analog_gain(self.currentConfiguration.analog_gain)

        # set illumination
        if self.control_illumination:
            self.set_illumination(
                self.currentConfiguration.illumination_source, self.currentConfiguration.illumination_intensity
            )

        # restart live
        if self.is_live is True:
            if self.control_illumination:
                self.turn_on_illumination()
            self.timer_trigger.start()

    def get_trigger_mode(self):
        return self.trigger_mode

    # slot
    def on_new_frame(self):
        if self.fps_trigger <= 5:
            if self.control_illumination and self.illumination_on == True:
                self.turn_off_illumination()

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling / 100

    def reset_strobe_arugment(self):
        # re-calculate the strobe_delay_us value
        try:
            self.camera.calculate_hardware_trigger_arguments()
        except AttributeError:
            pass
        self.microcontroller.set_strobe_delay_us(self.camera.strobe_delay_us)


class SlidePositionControlWorker(QObject):

    finished = Signal()
    signal_stop_live = Signal()
    signal_resume_live = Signal()

    def __init__(self, slidePositionController, stage: AbstractStage, home_x_and_y_separately=False):
        QObject.__init__(self)
        self.slidePositionController = slidePositionController
        self.stage = stage
        self.liveController = self.slidePositionController.liveController
        self.home_x_and_y_separately = home_x_and_y_separately

    def move_to_slide_loading_position(self):
        was_live = self.liveController.is_live
        if was_live:
            self.signal_stop_live.emit()

        # retract z
        self.slidePositionController.z_pos = self.stage.get_pos().z_mm  # zpos at the beginning of the scan
        self.stage.move_z_to(OBJECTIVE_RETRACTED_POS_MM, blocking=False)
        self.stage.wait_for_idle(SLIDE_POTISION_SWITCHING_TIMEOUT_LIMIT_S)

        print("z retracted")
        self.slidePositionController.objective_retracted = True

        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # So we can home without issue, set our limits to something large.  Then later reset them back to
            # the safe values.
            a_large_limit_mm = 100
            self.stage.set_limits(
                x_pos_mm=a_large_limit_mm,
                x_neg_mm=-a_large_limit_mm,
                y_pos_mm=a_large_limit_mm,
                y_neg_mm=-a_large_limit_mm,
            )

            # home for the first time
            if self.slidePositionController.homing_done == False:
                print("running homing first")
                timestamp_start = time.time()
                # x needs to be at > + 20 mm when homing y
                self.stage.move_x(20)
                self.stage.home(y=True)
                self.stage.home(x=True)

                self.slidePositionController.homing_done = True
            # homing done previously
            else:
                self.stage.move_x_to(20)
                self.stage.move_y_to(SLIDE_POSITION.LOADING_Y_MM)
                self.stage.move_x_to(SLIDE_POSITION.LOADING_X_MM)
            # set limits again
            self.stage.set_limits(
                x_pos_mm=self.stage.get_config().X_AXIS.MAX_POSITION,
                x_neg_mm=self.stage.get_config().X_AXIS.MIN_POSITION,
                y_pos_mm=self.stage.get_config().Y_AXIS.MAX_POSITION,
                y_neg_mm=self.stage.get_config().Y_AXIS.MIN_POSITION,
            )
        else:

            # for glass slide
            if self.slidePositionController.homing_done == False or SLIDE_POTISION_SWITCHING_HOME_EVERYTIME:
                if self.home_x_and_y_separately:
                    self.stage.home(x=True)
                    self.stage.move_x_to(SLIDE_POSITION.LOADING_X_MM)

                    self.stage.home(y=True)
                    self.stage.move_y_to(SLIDE_POSITION.LOADING_Y_MM)
                else:
                    self.stage.home(x=True, y=True)

                    self.stage.move_x_to(SLIDE_POSITION.LOADING_X_MM)
                    self.stage.move_y_to(SLIDE_POSITION.LOADING_Y_MM)
                self.slidePositionController.homing_done = True
            else:
                self.stage.move_y_to(SLIDE_POSITION.LOADING_Y_MM)
                self.stage.move_x_to(SLIDE_POSITION.LOADING_X_MM)

        if was_live:
            self.signal_resume_live.emit()

        self.slidePositionController.slide_loading_position_reached = True
        self.finished.emit()

    def move_to_slide_scanning_position(self):
        was_live = self.liveController.is_live
        if was_live:
            self.signal_stop_live.emit()

        # move to position
        # for well plate
        if self.slidePositionController.is_for_wellplate:
            # home for the first time
            if self.slidePositionController.homing_done == False:
                timestamp_start = time.time()

                # x needs to be at > + 20 mm when homing y
                self.stage.move_x_to(20)
                # home y
                self.stage.home(y=True)
                # home x
                self.stage.home(x=True)
                self.slidePositionController.homing_done = True

                # move to scanning position
                self.stage.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                self.stage.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
            else:
                self.stage.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                self.stage.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
        else:
            if self.slidePositionController.homing_done == False or SLIDE_POTISION_SWITCHING_HOME_EVERYTIME:
                if self.home_x_and_y_separately:
                    self.stage.home(y=True)

                    self.stage.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)

                    self.stage.home(x=True)
                    self.stage.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                else:
                    self.stage.home(x=True, y=True)

                    self.stage.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
                    self.stage.move_x_to(SLIDE_POSITION.SCANNING_X_MM)
                self.slidePositionController.homing_done = True
            else:
                self.stage.move_y_to(SLIDE_POSITION.SCANNING_Y_MM)
                self.stage.move_x_to(SLIDE_POSITION.SCANNING_X_MM)

        # restore z
        if self.slidePositionController.objective_retracted:
            # NOTE(imo): We want to move backlash compensation down to the firmware level.  Also, before the Stage
            # migration, we only compensated for backlash in the case that we were using PID control.  Since that
            # info isn't plumbed through yet (or ever from now on?), we just always compensate now.  It doesn't hurt
            # in the case of not needing it, except that it's a little slower because we need 2 moves.
            mm_to_clear_backlash = self.stage.get_config().Z_AXIS.convert_to_real_units(
                max(160, 20 * self.stage.get_config().Z_AXIS.MICROSTEPS_PER_STEP)
            )
            self.stage.move_z_to(self.slidePositionController.z_pos - mm_to_clear_backlash)
            self.stage.move_z_to(self.slidePositionController.z_pos)
            self.slidePositionController.objective_retracted = False
            print("z position restored")

        if was_live:
            self.signal_resume_live.emit()

        self.slidePositionController.slide_scanning_position_reached = True
        self.finished.emit()


class SlidePositionController(QObject):

    signal_slide_loading_position_reached = Signal()
    signal_slide_scanning_position_reached = Signal()
    signal_clear_slide = Signal()

    def __init__(self, stage: AbstractStage, liveController, is_for_wellplate=False):
        QObject.__init__(self)
        self.stage = stage
        self.liveController = liveController
        self.slide_loading_position_reached = False
        self.slide_scanning_position_reached = False
        self.homing_done = False
        self.is_for_wellplate = is_for_wellplate
        self.retract_objective_before_moving = RETRACT_OBJECTIVE_BEFORE_MOVING_TO_LOADING_POSITION
        self.objective_retracted = False
        self.thread = None

    def move_to_slide_loading_position(self):
        # create a QThread object
        self.thread = QThread()
        # create a worker object
        self.slidePositionControlWorker = SlidePositionControlWorker(self, self.stage)
        # move the worker to the thread
        self.slidePositionControlWorker.moveToThread(self.thread)
        # connect signals and slots
        self.thread.started.connect(self.slidePositionControlWorker.move_to_slide_loading_position)
        self.slidePositionControlWorker.signal_stop_live.connect(self.slot_stop_live, type=Qt.BlockingQueuedConnection)
        self.slidePositionControlWorker.signal_resume_live.connect(
            self.slot_resume_live, type=Qt.BlockingQueuedConnection
        )
        self.slidePositionControlWorker.finished.connect(self.signal_slide_loading_position_reached.emit)
        self.slidePositionControlWorker.finished.connect(self.slidePositionControlWorker.deleteLater)
        self.slidePositionControlWorker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.quit)
        # self.slidePositionControlWorker.finished.connect(self.threadFinished,type=Qt.BlockingQueuedConnection)
        # start the thread
        self.thread.start()

    def move_to_slide_scanning_position(self):
        # create a QThread object
        self.thread = QThread()
        # create a worker object
        self.slidePositionControlWorker = SlidePositionControlWorker(self, self.stage)
        # move the worker to the thread
        self.slidePositionControlWorker.moveToThread(self.thread)
        # connect signals and slots
        self.thread.started.connect(self.slidePositionControlWorker.move_to_slide_scanning_position)
        self.slidePositionControlWorker.signal_stop_live.connect(self.slot_stop_live, type=Qt.BlockingQueuedConnection)
        self.slidePositionControlWorker.signal_resume_live.connect(
            self.slot_resume_live, type=Qt.BlockingQueuedConnection
        )
        self.slidePositionControlWorker.finished.connect(self.signal_slide_scanning_position_reached.emit)
        self.slidePositionControlWorker.finished.connect(self.slidePositionControlWorker.deleteLater)
        self.slidePositionControlWorker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.quit)
        # self.slidePositionControlWorker.finished.connect(self.threadFinished,type=Qt.BlockingQueuedConnection)
        # start the thread
        print("before thread.start()")
        self.thread.start()
        self.signal_clear_slide.emit()

    def slot_stop_live(self):
        self.liveController.stop_live()

    def slot_resume_live(self):
        self.liveController.start_live()


class AutofocusWorker(QObject):

    finished = Signal()
    image_to_display = Signal(np.ndarray)
    # signal_current_configuration = Signal(Configuration)

    def __init__(self, autofocusController):
        QObject.__init__(self)
        self.autofocusController = autofocusController

        self.camera = self.autofocusController.camera
        self.microcontroller = self.autofocusController.microcontroller
        self.stage = self.autofocusController.stage
        self.liveController = self.autofocusController.liveController

        self.N = self.autofocusController.N
        self.deltaZ = self.autofocusController.deltaZ

        self.crop_width = self.autofocusController.crop_width
        self.crop_height = self.autofocusController.crop_height

    def run(self):
        self.run_autofocus()
        self.finished.emit()

    def wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(SLEEP_TIME_S)

    def run_autofocus(self):
        # @@@ to add: increase gain, decrease exposure time
        # @@@ can move the execution into a thread - done 08/21/2021
        focus_measure_vs_z = [0] * self.N
        focus_measure_max = 0

        z_af_offset = self.deltaZ * round(self.N / 2)

        # maneuver for achiving uniform step size and repeatability when using open-loop control
        # can be moved to the firmware
        mm_to_clear_backlash = self.stage.get_config().Z_AXIS.convert_to_real_units(
            max(160, 20 * self.stage.get_config().Z_AXIS.MICROSTEPS_PER_STEP)
        )

        self.stage.move_z(-mm_to_clear_backlash - z_af_offset)
        self.stage.move_z(mm_to_clear_backlash)

        steps_moved = 0
        for i in range(self.N):
            self.stage.move_z(self.deltaZ)
            steps_moved = steps_moved + 1
            # trigger acquisition (including turning on the illumination) and read frame
            if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
                self.liveController.turn_on_illumination()
                self.wait_till_operation_is_completed()
                self.camera.send_trigger()
                image = self.camera.read_frame()
            elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
                if "Fluorescence" in self.liveController.currentConfiguration.name and ENABLE_NL5 and NL5_USE_DOUT:
                    self.camera.image_is_ready = False  # to remove
                    self.microscope.nl5.start_acquisition()
                    image = self.camera.read_frame(reset_image_ready_flag=False)
                else:
                    self.microcontroller.send_hardware_trigger(
                        control_illumination=True, illumination_on_time_us=self.camera.exposure_time * 1000
                    )
                    image = self.camera.read_frame()
            if image is None:
                continue
            # tunr of the illumination if using software trigger
            if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
                self.liveController.turn_off_illumination()

            image = utils.crop_image(image, self.crop_width, self.crop_height)
            image = utils.rotate_and_flip_image(
                image, rotate_image_angle=self.camera.rotate_image_angle, flip_image=self.camera.flip_image
            )
            self.image_to_display.emit(image)
            # image_to_display = utils.crop_image(image,round(self.crop_width* self.liveController.display_resolution_scaling), round(self.crop_height* self.liveController.display_resolution_scaling))

            QApplication.processEvents()
            timestamp_0 = time.time()
            focus_measure = utils.calculate_focus_measure(image, FOCUS_MEASURE_OPERATOR)
            timestamp_1 = time.time()
            print("             calculating focus measure took " + str(timestamp_1 - timestamp_0) + " second")
            focus_measure_vs_z[i] = focus_measure
            print(i, focus_measure)
            focus_measure_max = max(focus_measure, focus_measure_max)
            if focus_measure < focus_measure_max * AF.STOP_THRESHOLD:
                break

        QApplication.processEvents()

        # maneuver for achiving uniform step size and repeatability when using open-loop control
        # TODO(imo): The backlash handling should be done at a lower level.  For now, do backlash compensation no matter if it makes sense to do or not (it is not harmful if it doesn't make sense)
        mm_to_clear_backlash = self.stage.get_config().Z_AXIS.convert_to_real_units(
            max(160, 20 * self.stage.get_config().Z_AXIS.MICROSTEPS_PER_STEP)
        )
        self.stage.move_z(-mm_to_clear_backlash - steps_moved * self.deltaZ)
        # determine the in-focus position
        idx_in_focus = focus_measure_vs_z.index(max(focus_measure_vs_z))
        self.stage.move_z(mm_to_clear_backlash + (idx_in_focus + 1) * self.deltaZ)

        QApplication.processEvents()

        # move to the calculated in-focus position
        if idx_in_focus == 0:
            print("moved to the bottom end of the AF range")
        if idx_in_focus == self.N - 1:
            print("moved to the top end of the AF range")


class AutoFocusController(QObject):

    z_pos = Signal(float)
    autofocusFinished = Signal()
    image_to_display = Signal(np.ndarray)

    def __init__(self, camera, stage: AbstractStage, liveController, microcontroller: Microcontroller):
        QObject.__init__(self)
        self.camera = camera
        self.stage = stage
        self.microcontroller = microcontroller
        self.liveController = liveController
        self.N = None
        self.deltaZ = None
        self.crop_width = AF.CROP_WIDTH
        self.crop_height = AF.CROP_HEIGHT
        self.autofocus_in_progress = False
        self.focus_map_coords = []
        self.use_focus_map = False

    def set_N(self, N):
        self.N = N

    def set_deltaZ(self, delta_z_um):
        self.deltaZ = delta_z_um / 1000

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def autofocus(self, focus_map_override=False):
        # TODO(imo): We used to have the joystick button wired up to autofocus, but took it out in a refactor.  It needs to be restored.
        if self.use_focus_map and (not focus_map_override):
            self.autofocus_in_progress = True

            self.stage.wait_for_idle(1.0)
            pos = self.stage.get_pos()

            # z here is in mm because that's how the navigation controller stores it
            target_z = utils.interpolate_plane(*self.focus_map_coords[:3], (pos.x_mm, pos.y_mm))
            print(f"Interpolated target z as {target_z} mm from focus map, moving there.")
            self.stage.move_z_to(target_z)
            self.autofocus_in_progress = False
            self.autofocusFinished.emit()
            return
        # stop live
        if self.liveController.is_live:
            self.was_live_before_autofocus = True
            self.liveController.stop_live()
        else:
            self.was_live_before_autofocus = False

        # temporarily disable call back -> image does not go through streamHandler
        if self.camera.callback_is_enabled:
            self.callback_was_enabled_before_autofocus = True
            self.camera.disable_callback()
        else:
            self.callback_was_enabled_before_autofocus = False

        self.autofocus_in_progress = True

        # create a QThread object
        try:
            if self.thread.isRunning():
                print("*** autofocus thread is still running ***")
                self.thread.terminate()
                self.thread.wait()
                print("*** autofocus threaded manually stopped ***")
        except:
            pass
        self.thread = QThread()
        # create a worker object
        self.autofocusWorker = AutofocusWorker(self)
        # move the worker to the thread
        self.autofocusWorker.moveToThread(self.thread)
        # connect signals and slots
        self.thread.started.connect(self.autofocusWorker.run)
        self.autofocusWorker.finished.connect(self._on_autofocus_completed)
        self.autofocusWorker.finished.connect(self.autofocusWorker.deleteLater)
        self.autofocusWorker.finished.connect(self.thread.quit)
        self.autofocusWorker.image_to_display.connect(self.slot_image_to_display)
        self.thread.finished.connect(self.thread.quit)
        # start the thread
        self.thread.start()

    def _on_autofocus_completed(self):
        # re-enable callback
        if self.callback_was_enabled_before_autofocus:
            self.camera.enable_callback()

        # re-enable live if it's previously on
        if self.was_live_before_autofocus:
            self.liveController.start_live()

        # emit the autofocus finished signal to enable the UI
        self.autofocusFinished.emit()
        QApplication.processEvents()
        print("autofocus finished")

        # update the state
        self.autofocus_in_progress = False

    def slot_image_to_display(self, image):
        self.image_to_display.emit(image)

    def wait_till_autofocus_has_completed(self):
        while self.autofocus_in_progress == True:
            QApplication.processEvents()
            time.sleep(0.005)
        print("autofocus wait has completed, exit wait")

    def set_focus_map_use(self, enable):
        if not enable:
            print("Disabling focus map.")
            self.use_focus_map = False
            return
        if len(self.focus_map_coords) < 3:
            print("Not enough coordinates (less than 3) for focus map generation, disabling focus map.")
            self.use_focus_map = False
            return
        x1, y1, _ = self.focus_map_coords[0]
        x2, y2, _ = self.focus_map_coords[1]
        x3, y3, _ = self.focus_map_coords[2]

        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if detT == 0:
            print("Your 3 x-y coordinates are linear, cannot use to interpolate, disabling focus map.")
            self.use_focus_map = False
            return

        if enable:
            print("Enabling focus map.")
            self.use_focus_map = True

    def clear_focus_map(self):
        self.focus_map_coords = []
        self.set_focus_map_use(False)

    def gen_focus_map(self, coord1, coord2, coord3):
        """
        Navigate to 3 coordinates and get your focus-map coordinates
        by autofocusing there and saving the z-values.
        :param coord1-3: Tuples of (x,y) values, coordinates in mm.
        :raise: ValueError if coordinates are all on the same line
        """
        x1, y1 = coord1
        x2, y2 = coord2
        x3, y3 = coord3
        detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if detT == 0:
            raise ValueError("Your 3 x-y coordinates are linear")

        self.focus_map_coords = []

        for coord in [coord1, coord2, coord3]:
            print(f"Navigating to coordinates ({coord[0]},{coord[1]}) to sample for focus map")
            self.stage.move_x_to(coord[0])
            self.stage.move_y_to(coord[1])

            print("Autofocusing")
            self.autofocus(True)
            self.wait_till_autofocus_has_completed()
            pos = self.stage.get_pos()

            print(f"Adding coordinates ({pos.x_mm},{pos.y_mm},{pos.z_mm}) to focus map")
            self.focus_map_coords.append((pos.x_mm, pos.y_mm, pos.z_mm))

        print("Generated focus map.")

    def add_current_coords_to_focus_map(self):
        if len(self.focus_map_coords) >= 3:
            print("Replacing last coordinate on focus map.")
        self.stage.wait_for_idle(timeout_s=0.5)
        print("Autofocusing")
        self.autofocus(True)
        self.wait_till_autofocus_has_completed()
        pos = self.stage.get_pos()
        x = pos.x_mm
        y = pos.y_mm
        z = pos.z_mm
        if len(self.focus_map_coords) >= 2:
            x1, y1, _ = self.focus_map_coords[0]
            x2, y2, _ = self.focus_map_coords[1]
            x3 = x
            y3 = y

            detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if detT == 0:
                raise ValueError(
                    "Your 3 x-y coordinates are linear. Navigate to a different coordinate or clear and try again."
                )
        if len(self.focus_map_coords) >= 3:
            self.focus_map_coords.pop()
        self.focus_map_coords.append((x, y, z))
        print(f"Added triple ({x},{y},{z}) to focus map")


class MultiPointWorker(QObject):

    finished = Signal()
    image_to_display = Signal(np.ndarray)
    spectrum_to_display = Signal(np.ndarray)
    image_to_display_multi = Signal(np.ndarray, int)
    signal_current_configuration = Signal(Configuration)
    signal_register_current_fov = Signal(float, float)
    signal_detection_stats = Signal(object)
    signal_update_stats = Signal(object)
    signal_z_piezo_um = Signal(float)
    napari_layers_init = Signal(int, int, object)
    napari_layers_update = Signal(np.ndarray, float, float, int, str)  # image, x_mm, y_mm, k, channel
    napari_rtp_layers_update = Signal(np.ndarray, str)
    signal_acquisition_progress = Signal(int, int, int)
    signal_region_progress = Signal(int, int)

    def __init__(self, multiPointController):
        QObject.__init__(self)
        self.multiPointController = multiPointController
        self._log = squid.logging.get_logger(__class__.__name__)
        self.signal_update_stats.connect(self.update_stats)
        self.start_time = 0
        if DO_FLUORESCENCE_RTP:
            self.processingHandler = multiPointController.processingHandler
        self.camera = self.multiPointController.camera
        self.microcontroller = self.multiPointController.microcontroller
        self.usb_spectrometer = self.multiPointController.usb_spectrometer
        self.stage: squid.abc.AbstractStage = self.multiPointController.stage
        self.liveController = self.multiPointController.liveController
        self.autofocusController = self.multiPointController.autofocusController
        self.configurationManager = self.multiPointController.configurationManager
        self.NX = self.multiPointController.NX
        self.NY = self.multiPointController.NY
        self.NZ = self.multiPointController.NZ
        self.Nt = self.multiPointController.Nt
        self.deltaX = self.multiPointController.deltaX
        self.deltaY = self.multiPointController.deltaY
        self.deltaZ = self.multiPointController.deltaZ
        self.dt = self.multiPointController.deltat
        self.do_autofocus = self.multiPointController.do_autofocus
        self.do_reflection_af = self.multiPointController.do_reflection_af
        self.crop_width = self.multiPointController.crop_width
        self.crop_height = self.multiPointController.crop_height
        self.display_resolution_scaling = self.multiPointController.display_resolution_scaling
        self.counter = self.multiPointController.counter
        self.experiment_ID = self.multiPointController.experiment_ID
        self.base_path = self.multiPointController.base_path
        self.selected_configurations = self.multiPointController.selected_configurations
        self.use_piezo = self.multiPointController.use_piezo
        self.detection_stats = {}
        self.async_detection_stats = {}
        self.timestamp_acquisition_started = self.multiPointController.timestamp_acquisition_started
        self.time_point = 0
        self.af_fov_count = 0
        self.num_fovs = 0
        self.total_scans = 0
        self.scan_region_fov_coords_mm = self.multiPointController.scan_region_fov_coords_mm.copy()
        self.scan_region_coords_mm = self.multiPointController.scan_region_coords_mm
        self.scan_region_names = self.multiPointController.scan_region_names
        self.z_stacking_config = self.multiPointController.z_stacking_config  # default 'from bottom'
        self.z_range = self.multiPointController.z_range

        self.microscope = self.multiPointController.parent
        self.performance_mode = self.microscope.performance_mode

        try:
            self.model = self.microscope.segmentation_model
        except:
            pass
        self.crop = SEGMENTATION_CROP

        self.t_dpc = []
        self.t_inf = []
        self.t_over = []

        if USE_NAPARI_FOR_MULTIPOINT:
            self.init_napari_layers = False

        self.count = 0

        self.merged_image = None
        self.image_count = 0

        # autofocus settings
        self.focusCount = 140
        self.numFrames = 1
        self.zRetract = -500
        self.focusChan = 'Fluorescence 405 nm Ex'
        self.stitchMode = 'MIP' # MIP or best focus
        self.autofocusDiffs = [] # record autofocus differences in um
        self.zOffset = 0 # um offset
        self.fillBoundary = False # large scan tiling
        self.zPlaneBestFit = [] # best fit z plane. Coefs a, b, c
        self.localPath = '' # local path for saving images
        self.registerToPrevTile = False # register to previous tile
        self.coordAlign = np.nan
        self.yAlign = np.nan
        self.xAlign = np.nan
        # self.prevTileName = '' # previous tile files
        self.frameLength = 0 # frame length in mm
        self.overlapFrac = 0.15 # overlap fraction
        self.NA = 22 # numerical aperture
        self.chanSelect = {} # channel selection
        self.skipTile = False # for skipping current tile if blank
        self.z_piezo_um = OBJECTIVE_PIEZO_HOME_UM
        self.spreadOil = False # spread oil

    def update_stats(self, new_stats):
        self.count += 1
        self._log.info("stats", self.count)
        for k in new_stats.keys():
            try:
                self.detection_stats[k] += new_stats[k]
            except:
                self.detection_stats[k] = 0
                self.detection_stats[k] += new_stats[k]
        if "Total RBC" in self.detection_stats and "Total Positives" in self.detection_stats:
            self.detection_stats["Positives per 5M RBC"] = 5e6 * (
                self.detection_stats["Total Positives"] / self.detection_stats["Total RBC"]
            )
        self.signal_detection_stats.emit(self.detection_stats)

    def update_use_piezo(self, value):
        self.use_piezo = value
        self._log.info(f"MultiPointWorker: updated use_piezo to {value}")

    def run(self):
        """Main run method that handles multi-sample acquisition from UI sample list"""
        try:
            # Check if we have a sample list from the UI
            if hasattr(self.multiPointController, 'sample_list') and self.multiPointController.sample_list:
                print(f"Running multi-sample acquisition with {len(self.multiPointController.sample_list)} samples")
                
                # Loop through each sample in the list
                for sample_idx, sample_config in enumerate(self.multiPointController.sample_list):
                    print(f"\n=== Processing Sample {sample_idx + 1}/{len(self.multiPointController.sample_list)} ===")
                    
                    # Set up exposure times for this sample
                    if 'selected_channels' in sample_config:
                        self.chanSelect = sample_config['selected_channels']
                    else:
                        self.chanSelect = 'UserSelected'

                    # Set location list for this sample
                    if 'location_list' in sample_config:
                        self.multiPointController.location_list = sample_config['location_list']
                    
                    # Set location IDs for this sample
                    if 'location_ids' in sample_config:
                        self.multiPointController.location_ids = sample_config['location_ids']

                    # # DEBUGGING: Print the sample configuration
                    # print(f"Sample {sample_idx + 1} configuration: {sample_config}")
                    
                    # Check for missing required values
                    required_params = [
                        'nx', 'ny', 'nz', 'dz', 'base_path', 'focusCount', 'numFrames', 
                        'zRetract', 'focusChan', 'fillBoundary', 'stitchMode', 
                        'initialOffset', 'spreadOil', 'registerToPrevTile', 
                        'objective', 'overlapFrac'
                    ]
                    
                    missing_params = []
                    for param in required_params:
                        if param not in sample_config or sample_config[param] is None:
                            missing_params.append(param)
                    
                    if missing_params:
                        error_msg = f"Sample {sample_idx + 1} is missing required parameters: {', '.join(missing_params)}"
                        print(f"ERROR: {error_msg}")
                        raise ValueError(error_msg)
                    
                    # Apply sample configuration using setExpParams
                    self.setExpParams(
                        NX=sample_config['nx'],
                        NY=sample_config['ny'], 
                        NZ=sample_config['nz'],
                        deltaZ=sample_config['dz'],
                        basePath=sample_config['base_path'],
                        focusCount=sample_config['focusCount'],
                        numFrames=sample_config['numFrames'],
                        zRetract=sample_config['zRetract'],
                        focusChan=sample_config['focusChan'],
                        fillBoundary=sample_config['fillBoundary'],
                        stitchMode=sample_config['stitchMode'],
                        initialOffset=sample_config['initialOffset'],
                        spreadOil=sample_config['spreadOil'],
                        registerToPrevTile=sample_config['registerToPrevTile'],
                        objective=sample_config['objective'],
                        overlapFrac=sample_config['overlapFrac'],
                        timePoint=sample_idx  # Use sample index as time point
                    )
                        
                    print(f"Sample {sample_idx + 1} acquisition completed")
                
                print("Multi-sample acquisition completed successfully")
                
        except Exception as e:
            print(f"Error in multi-sample acquisition: {e}")
            raise e
        finally:
            # stop/disconnect pump at finish
            if self.multiPointController.pump is not None: 
                self.multiPointController.pump.disconnect()   

    def setExpParams(self, NX, NY, NZ, deltaZ, basePath, focusCount = 100, 
                     numFrames = 1, zRetract = -5000, focusChan = 'Fluorescence 405 nm Ex', fillBoundary = False,
                     Nt = 1, timePoint = 0, stitchMode = 'MIP', initialOffset = 0, spreadOil = False, 
                     registerToPrevTile = False, objective = 60, overlapFrac = 0.15, 
                     blankThresh = -1, ): # set experimental parameters for each sample
        
        ### DEBUGGING: Print the parameters being set
        print(f"DEBUG: Setting experimental parameters:\n"
              f"NX: {NX}, NY: {NY}, NZ: {NZ}, deltaZ: {deltaZ}, basePath: {basePath},\n"
              f"focusCount: {focusCount}, numFrames: {numFrames}, zRetract: {zRetract},\n"
              f"focusChan: {focusChan}, fillBoundary: {fillBoundary}, stitchMode: {stitchMode},\n"
              f"initialOffset: {initialOffset}, spreadOil: {spreadOil},\n"
              f"registerToPrevTile: {registerToPrevTile}, objective: {objective}, overlapFrac: {overlapFrac},\n"
              f"blankThresh: {blankThresh}\n"
              )
        # print out all channel names and exposure times
        for chanName, exposure in self.chanSelect.items():
            print(f"Channel: {chanName}, Exposure Time: {exposure}")
        return

        # print('Setting experimental params...')
        chanSelect = self.chanSelect
        if zRetract > 0:
            zRetract = -zRetract # ensure negative
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.Nt = Nt
        self.deltaX = (1 - overlapFrac) * self.NA / objective / np.sqrt(2) # mm 
        self.deltaY = self.deltaX # mm 
        self.overlapFrac = overlapFrac
        self.deltaZ = deltaZ # um 
        self.stitchMode = stitchMode
        self.registerToPrevTile = registerToPrevTile
        if self.registerToPrevTile:
            # ask user to manually select which tile to align
            coordAlign, yAlign, xAlign = input('Enter the tile to align to (Coord, Y, X): ').split(',')
            self.coordAlign = int(coordAlign)
            self.yAlign = int(yAlign)
            self.xAlign = int(xAlign)

        # set objective lens resolution
        self.frameLength = 22 / objective / np.sqrt(2) # mm of FOV

        # blank thresholds for tile skipping
        # if user calibrated blank and sample areas, use skipping
        if blankThresh > 0:
            self.multiPointController.blankThresh = blankThresh
            print('Blank threshold set to', blankThresh)
            if self.multiPointController.sampleThresh <= 0: # assign based on blank
                self.multiPointController.sampleThresh = self.multiPointController.blankThresh * 5
                print('Sample threshold set to', self.multiPointController.sampleThresh)
            self.multiPointController.skipBlank = True

        # check if existing experiment ID fodler already
        # reformat path to linux if necessary
        basePath = basePath.replace('\\','/') # windows to Linux
        basePath = Path(basePath)
        if basePath.parts[0].startswith('Y:'): # not linux
            basePath = self.reformatPathWindowsToLinux(basePath)
            print('Reformatted base path to', basePath)
            assert basePath.exists(), 'Path does not exist: ' + str(basePath)

        folder = [f for f in Path(basePath).glob('*') if f.is_dir() and f.stem.startswith('_')]
        if len(folder) == 0:
            self.experiment_ID = self.multiPointController.experiment_ID # assign original timestamp when start button pressed
        else: # use existing folder
            self.experiment_ID = folder[0].name

        self.time_point = timePoint
        self.timestamp_acquisition_started = self.multiPointController.timestamp_acquisition_started
        self.base_path = basePath
        # autofocus settings
        self.focusCount = focusCount
        self.numFrames = numFrames
        self.zRetract = zRetract
        self.focusChan = focusChan
        self.spreadOil = spreadOil
        self.multiPointController.base_path = self.base_path
        # set coords
        csvFiles = [f for f in Path(self.base_path).glob('*.csv')]
        assert len(csvFiles) > 0, 'No CSV files found for ' + str(self.base_path)
        csvFiles.sort()
        self.multiPointController.location_list = pd.read_csv(csvFiles[-1])[['x (mm)', 'y (mm)', 'z (um)']].values # latest coords
        # offset initial Z
        # self.multiPointController.location_list[:,2] = self.multiPointController.location_list[:,2] + \
        #     initialOffset / 1000 # convert to mm
        self.zOffset = initialOffset
        
        # use coords as single points or fill in the boundary (stitching)
        self.fillBoundary = fillBoundary
        if self.fillBoundary: # use stitching
            # compute best fit z plane for tilted samples
            self.best_fit_z_plane(self.multiPointController.location_list) # XYZ
            # compute center point to center tiles around and ensure tiles cover entire area
            self.createCenterPoint(bounds = self.multiPointController.location_list, 
                                   columns = pd.read_csv(csvFiles[-1]).columns.tolist(),
                              stepY = self.deltaY, 
                              stepX = self.deltaX) # fill in coords for stitching
            
        else: # discrete multipoint. 
            self.focusCount = 140 # use max autofocus range for well plate

        # create folder for experiment 
        current_path = os.path.join(self.base_path,self.experiment_ID,str(self.time_point))
        Path(current_path).mkdir(parents=True, exist_ok=True)

        # use user selected channels and exposure for this acquisition
        if chanSelect == 'UserSelected':
            chanSelect = self.multiPointController.selected_configurations

        else: # select channels via hard code
            # or apply selected channels for this sample acquisition
            # self.selected_configurations = self.multiPointController.selected_configurations
            configSelect = []
            for excite, exposure in chanSelect.items(): # each selected channel

                # skip if channel exposure set to 0
                if exposure == 0:
                    continue

                for ii, config in enumerate(self.configurationManager.configurations): # loop thru all possible channels

                    # print('Checking channel:', config.name, 'with exposure time:', config.exposure_time)

                    if excite in config.name:
                        # set the exposure time
                        config.exposure_time = exposure
                        configSelect.append(config)

                        # also update the XML file with new exposure time
                        self.configurationManager.update_configuration(configuration_id = config.id, attribute_name = 'ExposureTime', new_value = exposure)

                        print('Selected channel:', config.name, 'with exposure time:', config.exposure_time)
                        break
            self.selected_configurations = configSelect # assign all channels
            assert len(self.selected_configurations) > 0, 'No channels selected'

        # apply selected settings to multipoint controller too
        # configManagerThrowaway.write_configuration_selected(self.selected_configurations,os.path.join(self.base_path,self.experiment_ID)+"/configurations.xml") # save the configuration for the experiment
        # acquisition_parameters = {'dx(mm)':self.deltaX, 'Nx':self.NX, 'dy(mm)':self.deltaY, 'Ny':self.NY, 'dz(um)':self.deltaZ*1000,'Nz':self.NZ,'dt(s)':self.deltat,'Nt':self.Nt,'with AF':self.do_autofocus,'with reflection AF':self.do_reflection_af}
        self.multiPointController.NX = self.NX
        self.multiPointController.NY = self.NY
        self.multiPointController.NZ = self.NZ
        self.multiPointController.Nt = self.Nt
        self.multiPointController.set_deltaX(self.deltaX)
        self.multiPointController.set_deltaY(self.deltaY)
        self.multiPointController.set_deltaZ(self.deltaZ)
        self.multiPointController.selected_configurations = self.selected_configurations # exposure times not changed apparently
        self.multiPointController.configurationManager = self.configurationManager
        # self.multiPointController.experiment_ID = self.experiment_ID

        # # checking all configurations
        # for config in self.multiPointController.selected_configurations:
        #     print('Selected channel:', config.name, 'with exposure time:', config.exposure_time)

        # # write settings
        self.multiPointController.start_new_experiment(experiment_ID = self.experiment_ID, 
                                                       createFolder = False)
        # configManagerThrowaway = ConfigurationManager(self.configurationManager.config_filename)
        # # assign settings
        # for config in self.selected_configurations:
        #     configManagerThrowaway.update_configuration(configuration_id = config.id, attribute_name = 'ExposureTime', new_value = config.exposure_time)

        
        # self.configurationManager.write_configuration_selected(self.multiPointController.selected_configurations,
        #                                                     os.path.join(self.multiPointController.base_path, 
        #                                                                  self.multiPointController.experiment_ID)+"/configurations.xml") # save the configuration for the experiment

        # format dataframe
        dfParams = pd.DataFrame(columns = ['ID', 'Name', 'ExposureTime', 'AnalogGain', 'IlluminationSource', 'IlluminationIntensity', 'CameraSN', 'ZOffset', 'PixelFormat', '_PixelFormat_options', 'Selected'])
        for config in self.configurationManager.configurations: # all config

            selected = False
            for selected_config in self.selected_configurations:
                if config.name == selected_config.name:
                    selected = True
                    break

            dfSub = {'ID': config.id, 'Name': config.name, 'ExposureTime': config.exposure_time, 
                                        'AnalogGain': config.analog_gain, 'IlluminationSource': config.illumination_source, 
                                        'IlluminationIntensity': config.illumination_intensity, 
                                        'CameraSN': config.camera_sn, 'ZOffset': config.z_offset, 'PixelFormat': config.pixel_format, 
                                        '_PixelFormat_options': config._pixel_format_options, 
                                        'Selected': selected}
            dfSub = pd.DataFrame(dfSub, index=[0])
            dfParams = pd.concat([dfParams, dfSub], ignore_index=True)

        dfParams.to_csv(os.path.join(self.base_path,self.experiment_ID)+"/configurations.csv", index=False)

        # run acquisition
        self.acquireSingleSample() # start acquisition
            
    # navigate thru all coords. Wait at each step for user to spread oil
    def spreadOilAllCoords(self, startIdx = 0):
        # distThresh = 5 # mm
        cumDist = 0 # cumulative distance, mm
        for idx, coord in enumerate((tqdm(self.multiPointController.location_list))):
            if idx < startIdx:
                continue
            # print('Navigating to coord', idx+1, 'of', len(self.multiPointController.location_list))
            # retract z
            self.stage.move_z_to(0.1)
            self.stage.move_x_to(coord[0])
            self.stage.move_y_to(coord[1])
            # restore z coord
            self.stage.move_z_to(coord[2])
            
            # self.liveController.start_live()
            if idx > 0: # add distance traveled
                cumDist += np.sqrt((coord[0] - self.multiPointController.location_list[idx-1][0])**2 + \
                                   (coord[1] - self.multiPointController.location_list[idx-1][1])**2)
            
            if cumDist > 20: # mm
                input('Add oil and click enter to continue...')
                cumDist = 0 # reset distance
            # if idx > 0:
            #     if np.abs(coord[0] - self.multiPointController.location_list[idx-1][0] > distThresh) or \
            #         np.abs(coord[1] - self.multiPointController.location_list[idx-1][1]) > distThresh:
            #         input('Add oil and click enter to continue...')

            time.sleep(1)

        input('Oil spreading complete. Click enter to start acquisition...')
        self.promptUserManualControl() # for manual multipt adjustments
    
    # compute best fit z plane for large tile scans
    # Function to compute the best-fit z-plane given ZYX coordinates
    def best_fit_z_plane(self, zyx_coords):
        """
        Computes the best-fit z-plane (z = ax + by + c) for a set of points with ZYX coordinates.
        
        Args:
            zyx_coords: A numpy array of shape (n_points, 3), where each row contains [z, y, x].
            
        Returns:
            Tuple (a, b, c) corresponding to the plane equation: z = ax + by + c.
        """
        # Extract x, y, z coordinates
        z = zyx_coords[:, 2]
        y = zyx_coords[:, 1]
        x = zyx_coords[:, 0]
        
        # Prepare the input for fitting: X will be a (n_points, 2) matrix with x and y as columns
        X = np.column_stack((x, y))
        
        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, z)
        
        # Return the coefficients a, b and intercept c
        a, b = model.coef_
        c = model.intercept_
        self.zPlaneBestFit = [a, b, c]
        
        return a, b, c
    
    def createCenterPoint(self, bounds, columns, stepX, stepY): # fill in coordinates for stitching
        bounds = pd.DataFrame(bounds, columns = ['x (mm)', 'y (mm)', 'z (um)'])
        coords = bounds.copy(deep = True)
        coords.rename(columns={'x (mm)': 'X', 
                            'y (mm)': 'Y'}, inplace=True)

        # compute start (smallest) coords
        xMin = coords['X'].min()
        xMax = coords['X'].max()
        yMin = coords['Y'].min()
        yMax = coords['Y'].max()

        # compute center point
        xCenter = (xMin + xMax) / 2
        yCenter = (yMin + yMax) / 2

        # print('Center X is', xCenter, 'mm')
        # print('Center Y is', yCenter, 'mm')

        # normalize coordinates
        coords[['X', 'Y']] = coords[['X', 'Y']] - coords[['X', 'Y']].min()

        # # use max length and divide by step
        tileX = np.ceil(coords['X'].max() / stepX)
        tileY = np.ceil(coords['Y'].max() / stepY)
        print('Number of X tiles is', int(tileX))
        print('Number of Y tiles is', int(tileY))

        # # Export updated CSV file with only start coords

        # X, Y, Z, ID, i, j, k
        zCenter = coords['z (um)'].mean()
        startCoord = pd.DataFrame(columns = columns, data = [[xCenter, 
                                                                    yCenter, 
                                                                    zCenter,
                                                                    np.nan, 
                                                                    0, 
                                                                    0, 
                                                                    0]])

        # # export CSV file with single start coord
        # fileOut = csvPath.parent / '02 start center coord.csv'
        # assert not fileOut.exists(), 'File already exists'
        # startCoord.to_csv(fileOut, index=False)

        # reassign to start coord
        self.multiPointController.location_list = startCoord[['x (mm)', 'y (mm)', 'z (um)']].values # latest coords
        # reassign NX and NY
        self.NX = int(tileX)
        self.NY = int(tileY)
        return None
    
    # reformat Windows paths to Linux
    def reformatPathWindowsToLinux(self, windowsPath):

        # Linux has different parent path than windows. Reformat parent to match windows so user can run Fiji in windows
        path = Path(windowsPath).parts
        # find Desktop string match
        for idxCoskun, part in enumerate(path):
            if part.startswith('coskun-lab'):
                break
        # path = Path(*path[idxCoskun + 1:])
        # add first half of path
        path = Path('/home', 'cephla', 'Desktop', *path[idxCoskun:]).parts
        # remove duplicate entries
        ordered = []
        for part in path:
            if part not in ordered:
                ordered.append(part)
        linuxPath = Path(*ordered)

        return linuxPath
    
    # perform homing
    def home_all_axes(self):

        # home Z 
        print('home z')
        self.stage.home_z()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.stage.zero_z()

        # home XY, set zero and set software limit
        print('home xy')
        timestamp_start = time.time()
        # x needs to be at > + 20 mm when homing y
        self.stage.move_x(20) # to-do: add blocking code
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        # home y
        self.stage.home_y()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 100:
                print('y homing timeout, the program will exit')
                sys.exit(1)
        self.stage.zero_y()
        # home x
        self.stage.home_x()
        t0 = time.time()
        while self.microcontroller.is_busy():
            time.sleep(0.005)
            if time.time() - t0 > 100:
                print('x homing timeout, the program will exit')
                sys.exit(1)
        self.stage.zero_x()

        if USE_ZABER_EMISSION_FILTER_WHEEL:
            self.microscope.emission_filter_wheel.wait_for_homing_complete()

        self.stage.set_x_limit_pos_mm(SOFTWARE_POS_LIMIT.X_POSITIVE)
        self.stage.set_x_limit_neg_mm(SOFTWARE_POS_LIMIT.X_NEGATIVE)
        self.stage.set_y_limit_pos_mm(SOFTWARE_POS_LIMIT.Y_POSITIVE)
        self.stage.set_y_limit_neg_mm(SOFTWARE_POS_LIMIT.Y_NEGATIVE)
        self.stage.set_z_limit_pos_mm(SOFTWARE_POS_LIMIT.Z_POSITIVE)

        # move to scanning position
        # if doHomeXY == 'y':
        self.stage.move_x(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)
        self.stage.move_y(20)
        while self.microcontroller.is_busy():
            time.sleep(0.005)

    # reformat Linux home paths to Windows
    def reformatPathLinuxToWindows(self, linuxPath):
        linuxPath = Path(linuxPath)
        # check OS
        if platform.system() == 'Linux':
            # Linux has different parent path than windows. Reformat parent to match windows so user can run Fiji in windows
            path = linuxPath.parts
            # find Desktop string match
            for idxCoskun, part in enumerate(path):
                if part == 'Desktop':
                    break
            # path = Path(*path[idxCoskun + 1:])
            # add first half of path
            path = Path('Y:', 'coskun-lab', *path[idxCoskun + 1:]).parts
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
    
    # write processing command to text file and save to disk for user to run later
    def writeProcessCommandTxt(self, current_path):
        txtFile = Path(current_path).parent / 'process_images_command_Windows.txt'
        f = open(txtFile, 'w')
        # write windows command
        # reformat Linux path to Windows
        windowsPath = Path(r'Y:\coskun-lab\Nicky\02 Microscope Tools\Cephla Squid\octopi-research-master\Current\software\control\process_images_real_time.py')
        f.write(f'python "{str(windowsPath)}"')
        f.write(f' {urllib.parse.quote(str(self.reformatPathLinuxToWindows(current_path)))} {self.overlapFrac} {self.NZ} {self.NY} {self.NX} {self.numFrames} {self.scan_coordinates_mm.shape[0]} {self.stitchMode} ""')
        f.close()

    def acquireSingleSample(self): # image single sample and then multiple if necessary

        # self.home_all_axes() # homing
        # print('Finished homing')

        self.start_time = time.perf_counter_ns()
        if self.camera.is_streaming == False:
             self.camera.start_streaming()

        if self.multiPointController.location_list is None:
            # use scanCoordinates for well plates or regular multipoint scan
            if self.multiPointController.scanCoordinates!=None and self.multiPointController.scanCoordinates.get_selected_wells():
                # use scan coordinates for the scan
                self.scan_coordinates_mm = self.multiPointController.scanCoordinates.coordinates_mm
                self.scan_coordinates_name = self.multiPointController.scanCoordinates.name
                self.use_scan_coordinates = True
            else:
                # use the current position for the scan
                self.scan_coordinates_mm = [(self.stage.get_pos().x_mm,self.stage.get_pos().y_mm)]
                self.scan_coordinates_name = ['']
                self.use_scan_coordinates = False
        else:
            # use location_list specified by the multipoint controlller
            self.scan_coordinates_mm = self.multiPointController.location_list
            # Use location_ids if available, otherwise set to None
            if hasattr(self.multiPointController, 'location_ids') and self.multiPointController.location_ids is not None:
                self.scan_coordinates_name = self.multiPointController.location_ids
            else:
                self.scan_coordinates_name = None
            self.use_scan_coordinates = True

        # execute estimate remaining time and processing script in another terminal
        current_path = os.path.join(self.base_path,self.experiment_ID,str(self.time_point))
        self.localPath = Path(r'/home/cephla/Downloads') / Path(*Path(current_path).parts[-4:]) # local save faster
        self.localPath.mkdir(parents = True, exist_ok = True)
        command = f'python3 ./control/process_images_real_time.py {urllib.parse.quote(current_path)} \
            {self.overlapFrac} {self.NZ} {self.NY} {self.NX} {self.numFrames} {self.scan_coordinates_mm.shape[0]} \
                 {self.stitchMode} {urllib.parse.quote(str(self.localPath))}'
        os.system(f"gnome-terminal -e 'bash -c \"{command};bash\"'")

        # also record processing command to text file for later use
        self.writeProcessCommandTxt(current_path=current_path)

        while self.time_point < self.Nt:

            # check if abort acquisition has been requested
            if self.multiPointController.abort_acqusition_requested:
                break

            # run single time point. Use async methods
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.run_single_time_point())
            
            self.time_point = self.time_point + 1
            # continous acquisition
            if self.dt == 0:
                pass
            # timed acquisition
            else:
                # check if the aquisition has taken longer than dt or integer multiples of dt, if so skip the next time point(s)
                while time.time() > self.timestamp_acquisition_started + self.time_point*self.dt:
                    print('skip time point ' + str(self.time_point+1))
                    self.time_point = self.time_point+1
                # check if it has reached Nt
                if self.time_point == self.Nt:
                    break # no waiting after taking the last time point
                # wait until it's time to do the next acquisition
                while time.time() < self.timestamp_acquisition_started + self.time_point*self.dt:
                    if self.multiPointController.abort_acqusition_requested:
                        break
                    time.sleep(0.05)

        self.processingHandler.processing_queue.join()
        self.processingHandler.upload_queue.join()
        elapsed_time = time.perf_counter_ns()-self.start_time
        print("Time taken for acquisition/processing: "+str(elapsed_time/10**9))
        # self.finished.emit()

    # start pump at constant speed
    def startPumpConst(self, speed = 10): # uL/min
        if self.multiPointController.pump is not None:
            self.multiPointController.pump.set_infusion_rate(speed, 'UM')
            time.sleep(0.2)
            self.multiPointController.pump.start_infuse() # continuous infusion

    # purge pump: high vol, short time
    def purgePump(self, speed = 100, duration = 1): # uL/s, sec
        if self.multiPointController.pump is not None:
            self.multiPointController.pump.set_infusion_rate(speed, 'ul/s')
            time.sleep(0.2)
            self.multiPointController.pump.start_infuse() # continuous infusion
            time.sleep(duration)
            # reset to base speed
            self.startPumpConst()

    def wait_till_operation_is_completed(self):
        while self.microcontroller.is_busy():
            time.sleep(SLEEP_TIME_S)

    async def run_single_time_point(self):
        start = time.time()
        print(time.time())
        # disable joystick button action
        # self.stage.enable_joystick_button_action = False

        print('multipoint acquisition - time point ' + str(self.time_point+1))

        # # TESTING live controller toggle
        # success = False
        # while not success:
        #     self.liveController.turn_on_illumination()
        #     self.liveController.start_live()
        #     print('Started live')
        #     time.sleep(5)
        #     self.liveController.stop_live()
        #     self.liveController.turn_on_illumination()
        #     print('Stopped live')
        #     time.sleep(5)
        
        # for each time point, create a new folder
        current_path = os.path.join(self.base_path,self.experiment_ID,str(self.time_point))
        Path(current_path, parents = True, exist_ok=True)
        # get current files already imaged, if any
        imgFiles = [f for f in Path(current_path).glob('*.tiff')]

        # create a dataframe to save coordinates
        if IS_HCS:
            if self.use_piezo:
                self.coordinates_pd = pd.DataFrame(columns = ['well', 'i', 'j', 'k', 'x (mm)', 'y (mm)', 'z (um)', 'z_piezo (um)', 'time'])
            else:
                self.coordinates_pd = pd.DataFrame(columns = ['well', 'i', 'j', 'k', 'x (mm)', 'y (mm)', 'z (um)', 'time'])
        else:
            if self.use_piezo:
                self.coordinates_pd = pd.DataFrame(columns = ['i', 'j', 'k', 'x (mm)', 'y (mm)', 'z (um)', 'z_piezo (um)', 'time'])
            else:
                self.coordinates_pd = pd.DataFrame(columns = ['i', 'j', 'k', 'x (mm)', 'y (mm)', 'z (um)', 'time'])

        n_regions = len(self.scan_coordinates_mm)

        started = False # start imaging when the specified coordinate is reached
        for coordinate_id in range(n_regions): # each coord

            # check if coordinate was already imaged entirely
            lastX = self.NX - 1 if self.NY % 2 != 0 else 0  # odd or even based on scan direction
            lastFile = f'{coordinate_id}_{str(self.NY - 1)}_{str(lastX)}_{str(self.NZ-1)}_'
            fileNames = [f for f in imgFiles if f.stem.startswith(lastFile)]
            if len(fileNames) > 0 and coordinate_id != self.coordAlign:
                print('Coordinate', coordinate_id, 'already imaged')
                continue

            # spread oil for wellplate for remaining coords
            if not started and self.spreadOil: self.spreadOilAllCoords(startIdx=coordinate_id)

            coordiante_mm = self.scan_coordinates_mm[coordinate_id]
            print(coordiante_mm)

            if self.scan_coordinates_name is None:
                # flexible scan, use a sequencial ID
                coordiante_name = str(coordinate_id)
            else:
                coordiante_name = self.scan_coordinates_name[coordinate_id]
            
            if self.use_scan_coordinates:
                # move to the specified coordinate. First coord NOT already imaged
                self.stage.move_x_to(coordiante_mm[0]-self.deltaX*(self.NX-1)/2)
                self.stage.move_y_to(coordiante_mm[1]-self.deltaY*(self.NY-1)/2)

                # ask user to calibrate blank and sample for skipping if not already set
                if coordinate_id == 0 and self.multiPointController.skipBlank == False:

                    ans, timed_out = timedInput(prompt='Do you want to calibrate blank area for tile skipping? (y/n): ', 
                                                       timeout=10)
                    if timed_out:
                        print("Timed out!")
                        ans = 'n'
                    if ans == 'y':
                        self.stage.move_z_to(coordiante_mm[2]) # restore z coord
                        input('Calibrate blank area then press enter to continue...')
                    # if np.var(self.multiPointController.blankThresh) > 0 or np.var(self.multiPointController.sampleThresh) > 0:
                    if self.multiPointController.blankThresh > 0 or self.multiPointController.sampleThresh > 0:
                        self.multiPointController.skipBlank = True
                        if self.multiPointController.sampleThresh <= 0: # assign based on blank
                            self.multiPointController.sampleThresh = self.multiPointController.blankThresh * 5
                        print('Blank and sample areas successfully calibrated')

                    # move back to starting coordinate
                    self.stage.move_x_to(coordiante_mm[0]-self.deltaX*(self.NX-1)/2)
                    self.stage.move_y_to(coordiante_mm[1]-self.deltaY*(self.NY-1)/2)

                self.purgePump() # purge pump at start of each coord
                    
                # check if z is included in the coordinate
                if len(coordiante_mm) == 3:
                    if coordiante_mm[2] >= self.stage.get_pos().z_mm:
                        self.stage.move_z_to(coordiante_mm[2])
                        
                    else:
                        self.stage.move_z_to(coordiante_mm[2])
                        
                    self.purgePump() # purge pump after touching sample
                            
                else:
                    time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
                    if len(coordiante_mm) == 3:
                        time.sleep(SCAN_STABILIZATION_TIME_MS_Z/1000)
                    # add '_' to the coordinate name
                    coordiante_name = coordiante_name + '_'

            self.x_scan_direction = 1
            self.dx_usteps = 0 # accumulated x displacement
            self.dy_usteps = 0 # accumulated y displacement
            self.dz_usteps = 0 # accumulated z displacement
            z_pos = self.stage.get_pos().z_mm # zpos at the beginning of the scan

            # z stacking config
            if Z_STACKING_CONFIG == 'FROM TOP':
                self.deltaZ_usteps = -abs(self.deltaZ_usteps)

            if USE_NAPARI_FOR_MULTIPOINT or USE_NAPARI_FOR_TILED_DISPLAY:
                init_napari_layers = False

            # reset piezo to home position
            if self.use_piezo:
                self.microcontroller.set_piezo_um(self.z_piezo_um)
                if self.liveController.trigger_mode == TriggerMode.SOFTWARE: # for hardware trigger, delay is in waiting for the last row to start exposure
                    time.sleep(MULTIPOINT_PIEZO_DELAY_MS/1000)
                if MULTIPOINT_PIEZO_UPDATE_DISPLAY:
                    self.signal_z_piezo_um.emit(self.z_piezo_um)

            # along y
            for i in range(self.NY): # each Y tile

                self.FOV_counter = 0 # for AF, so that AF at the beginning of each new row

                # along x
                for j in range(self.NX): # each X tile

                    # check if image already acquired
                    lastFile = f'{coordinate_id}_{str(i)}_{str(j if self.x_scan_direction==1 else self.NX-1-j)}_{str(self.NZ-1)}_'
                    # print('Checking for ', lastFile)
                    fileNames = [f for f in imgFiles if f.stem.startswith(lastFile)]
                    if len(fileNames) > 0: # tile already acquired
                        if coordinate_id == self.coordAlign and i == self.yAlign and j == self.xAlign:
                            print(f'Reached user selected tile to align Coord {self.coordAlign} Y{str(self.yAlign)} X{str(self.xAlign)}:', lastFile)
                            self.registerToSelectedTile(current_path, coordinate_id, coordiante_name, i, j, self.localPath) # wait for user to align tile, then continue

                        print('Skipping coord:', coordinate_id, 'Y:', i, 'X:', j)
                        # self.prevTileName = lastFile # record previous tile name
                        
                    else: # tile not acquired yet. image this tile
                        if started == False: # first time starting
                            # turn on spinning disk if want confocal mode
                            print('Turning ON spinning disk...')
                            self.microscope.xlight.set_disk_motor_state(1)
                            
                            pause = 30 # sec
                            print('Pausing', pause, 'sec for oil lens refill') # pause for oil lens refill
                            self.purgePump() # refill
                            for _ in trange(pause): # print timer
                                time.sleep(1)
                            started = True

                            # # if resuming imaging, use registration/phase cross correlation to align to previous tile and resume imaging
                            # # first move back to last complete tile
                            # if self.registerToPrevTile: self.registerToLastCompleteTile(current_path, coordinate_id, coordiante_name, i, j, self.localPath)
                       
                        # use async calls to check for crashing
                        taskAcquire = asyncio.create_task(self.multipoint_custom_script_entry(self.time_point,current_path,coordinate_id,
                                                                                           coordiante_name,i,j, self.localPath))
                        await taskAcquire # wait for tile to finish

                    if self.NX > 1:
                        # move x
                        if j < self.NX - 1:
                            # self.stage.move_x_usteps(self.x_scan_direction*self.deltaX_usteps)
                            self.stage.move_x(self.x_scan_direction*self.deltaX)
                            
                            # time.sleep(SCAN_STABILIZATION_TIME_MS_X/1000)
                            # self.dx_usteps = self.dx_usteps + self.x_scan_direction*self.deltaX_usteps

                    # check for low disk space
                    self.checkLowDiskSpace(current_path)

                # finished X scan
                self.x_scan_direction = -self.x_scan_direction

                if self.NY > 1:
                    # move y
                    if i < self.NY - 1:
                        self.stage.move_y(self.deltaY)
                        # time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
                        # self.dy_usteps = self.dy_usteps + self.deltaY_usteps

                # at end of y row, purge pump again during tile scan
                if self.fillBoundary: 
                    self.purgePump()

            # finished XY scan
            if SHOW_TILED_PREVIEW and IS_HCS:
                self.stage.keep_scan_begin_position(self.stage.get_pos().x_mm, self.stage.get_pos().y_mm)

            if n_regions == 1:
                # only move to the start position if there's only one region in the scan
                if self.NY > 1:
                    # move y back
                    # self.stage.move_y_usteps(-self.deltaY_usteps*(self.NY-1))
                    self.stage.move_y(-self.deltaY*(self.NY-1))
                    
                    # time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
                    # self.dy_usteps = self.dy_usteps - self.deltaY_usteps*(self.NY-1)

                if SHOW_TILED_PREVIEW and not IS_HCS:
                    self.stage.keep_scan_begin_position(self.stage.get_pos().x_mm, self.stage.get_pos().y_mm)

                # move x back at the end of the scan
                if self.x_scan_direction == -1:
                    # self.stage.move_x_usteps(-self.deltaX_usteps*(self.NX-1))
                    self.stage.move_x(-self.deltaX*(self.NX-1))
                    
                    # time.sleep(SCAN_STABILIZATION_TIME_MS_X/1000)

                # # move z back
                # if self.stage.get_pid_control_flag(2) is False:
                #     _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                #     self.stage.microcontroller.move_z_to_usteps(z_pos - STAGE_MOVEMENT_SIGN_Z*_usteps_to_clear_backlash)
                    
                #     self.stage.move_z_usteps(_usteps_to_clear_backlash)
                    
                # else:
                #     self.stage.microcontroller.move_z_to_usteps(z_pos)
                    

        # # move any last remaining files
        # taskMoveFiles = asyncio.create_task(self.copyLocalFilesToNetwork(self.localPath, current_path))
        # await taskMoveFiles

        # finished region scan
        self.coordinates_pd.to_csv(os.path.join(current_path,'coordinates.csv'),index=False,header=True)
        # self.stage.enable_joystick_button_action = True
        print(time.time())
        print(time.time()-start)

    # ask user to manually trigger live view and align with previous complete tile. For some reason, the start live button causes bugs
    def promptUserManualControl(self):

        while True: 
            prompt = 'Align microscope manually. Click enter when finished. Press numbers (_,_) to trigger live view with different channels and durations: '
            chIdx = 1
            chanConfig = {}
            for excite, exposure in self.chanSelect.items(): # each selected channel
                # skip if channel exposure set to 0
                if exposure == 0:
                    continue
                for ii, config in enumerate(self.configurationManager.configurations): # loop thru all possible channels
                    if excite in config.name:
                        # print('Selected channel:', config.name, 'with exposure time:', config.exposure_time)
                        prompt += f'\n {chIdx} for {config.name}, '
                        chanConfig[chIdx] = config
                        chIdx += 1
                        break

            prompt += '\n'
            ans = input(prompt)
            if ans == '': # user presssed enter key when finished
                return
            idx, duration = ans.split(',')

            # numeric input. Keep running for set amount of time
            if idx.isnumeric() and duration.isnumeric():
                for ii in trange(int(duration)): # sec
                    self.acquire_camera_image(chanConfig[int(idx)])
                    # self.liveController.start_live()
                    time.sleep(1)
            
    # register to last complete tile to ensure alignment
    def registerToLastCompleteTile(self, current_path, coordinate_id, coordiante_name, i, j, localPath):
        movedX = False
        movedY = False

        if j == 0 and self.x_scan_direction == 1: # first x tile. move y back
            
            # self.stage.move_y_usteps(-self.deltaY_usteps)
            self.stage.move_y(-self.deltaY)
            
            time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
            movedY = True

        elif j == self.NX - 1 and self.x_scan_direction == -1: # first x tile in reverse. move y back

            # self.stage.move_y_usteps(-self.deltaY_usteps)
            self.stage.move_y(-self.deltaY)
            
            time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
            movedY = True

        else: # middle of x row. Move x back
            # self.stage.move_x_usteps(-1 * self.x_scan_direction*self.deltaX_usteps)
            self.stage.move_x(-1 * self.x_scan_direction*self.deltaX)
            
            # time.sleep(SCAN_STABILIZATION_TIME_MS_X/1000)
            movedX = True

        # or wait for user to adjust
        # ans = input('Register automatically (a) or manually (m)? ')
        # if ans == 'm':

        # input('Adjust microscope manually. and click enter to continue...')
        self.promptUserManualControl()
        
        # elif ans == 'a':
        #     pass
            # # find last complete tile image
            # print('Last complete tile is', self.prevTileName)
            # coord, y, x, _, _ = self.prevTileName.split('_')
            # # find image files for focus chan
            # fileNames = [f for f in Path(current_path).glob('*.tiff') if f.stem.startswith(f'{coord}_{y}_{x}_') and 
            #             self.focusChan.replace(' ', '_') in f.stem]
            # assert len(fileNames) == self.NZ, 'Missing files for last complete tile'
            # # read z stack
            # img = tf.TiffFile(fileNames[0])
            # dimY, dimX = img.pages[0].shape
            # dtype = img.pages[0].dtype
            # img.close()
            # lazy_read = dask.delayed(tf.imread)
            # zStack = []
            # for f in fileNames: # each z plane
            #     plane = lazy_read(f)
            #     plane = da.from_delayed(plane, shape=(dimY, dimX), dtype=dtype)
            #     zStack.append(plane) # each YX
            # zStack = da.stack(zStack, axis=0) # ZYX
            # prevMip = np.max(zStack, axis=0).compute() # MIP, YX
            # # convert to 8-bit
            # prevMip = (prevMip - prevMip.min()) / (prevMip.max() - prevMip.min()) * 255
            # prevMip = np.array(prevMip, dtype=np.uint8)

            # # find register current image to previous until it's within tolerance
            # registered = False
            # while not registered:

            #     print('Autofocus scanning current tile...')
            #     # find XY coords of current tile
            #     focusStep = 2 # um
            #     x = self.stage.get_pos().x_mm # current X coord
            #     y = self.stage.get_pos().y_mm # current Y coord
            #     a, b, c = self.zPlaneBestFit # Coefs a, b, c
            #     z = a*x + b*y + c # best fit plane equation
            #     print(f'Moving to interpolated Z coord {z} minus half of autofocus range {focusStep * self.focusCount / 2000} mm...')
            #     self.stage.move_z_to(z - focusStep * self.focusCount / 2000 + self.zOffset / 1000) # um to mm
            #     print('Running autofocus scan')
            #     # home piezo
            #     if self.use_piezo and self.z_piezo_um != OBJECTIVE_PIEZO_HOME_UM:
            #         self.movePiezo(OBJECTIVE_PIEZO_HOME_UM)
            #     _, focusScan = self.acquireAutofocusScan()
            #     focusMip = np.max(focusScan, axis=0) # MIP, YX
            #     # convert to 8-bit
            #     focusMip = (focusMip - focusMip.min()) / (focusMip.max() - focusMip.min()) * 255
            #     focusMip = np.array(focusMip, dtype=np.uint8)

            #     # print('focusMip type', focusMip.dtype, 'shape', focusMip.shape)
            #     # print('prevMip type', prevMip.dtype, 'shape', prevMip.shape)

            #     # compute registration with cv2. Estimate translation from homography matrix
            #     # shifts, _, _ = skimage.registration.phase_cross_correlation(reference_image = prevMip, moving_image = focusMip, normalization=None)
            #     _, H = self.Image_registration(prevMip, focusMip) # compute homography matrix
            #     # compute translation from homography matrix
            #     # Camera matrix (assuming identity for simplicity)
            #     K = np.eye(3)
            #     # Decompose the homography matrix
            #     _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
            #     # Extract the translation vector
            #     translation = Ts[0]  # There can be up to 4 solutions, choose the first one

            #     # shift microscope by translation
            #     yShift = translation[0][0]
            #     xShift = translation[1][0]
            #     print(f'Shifts are Y {yShift} and X {xShift} pixels...') # px

            #     if np.abs(yShift) / dimY < 0.05 and np.abs(xShift) / dimX < 0.05: # less than 5% shift left
            #         registered = True

            #     # flip directions to match microscope stage orientation
            #     # yShift = -yShift
            #     xShift = -xShift

            #     # convert shifts from px to mm
            #     yShift = yShift / dimY * self.frameLength # mm
            #     xShift = xShift / dimX * self.frameLength # mm
            #     print(f'Shifts are Y {yShift} and X {xShift} mm...') # mm

            #     # apply shifts to microscope
            #     self.stage.move_x_usteps(self.convert_mm_to_usteps(xShift))
            #     
            #     time.sleep(SCAN_STABILIZATION_TIME_MS_X/1000)
            #     self.stage.move_y_usteps(self.convert_mm_to_usteps(yShift))
            #     
            #     time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)

        # else:
        #     print('Invalid input. Exiting...')

        # move back to current tile after successful registration
        print('Registration successful. Moving back to current tile...')
        if movedY: # first x tile. move y back
            # self.stage.move_y_usteps(self.deltaY_usteps)
            self.stage.move_y(self.deltaY)
            
            # time.sleep(SCAN_STABILIZATION_TIME_MS_Y/1000)
            print('Moved to current Y tile...')

        if movedX: # middle of x row. Move x back
            # self.stage.move_x_usteps(self.x_scan_direction*self.deltaX_usteps)
            self.stage.move_x(self.x_scan_direction*self.deltaX)
            
            # time.sleep(SCAN_STABILIZATION_TIME_MS_X/1000)
            print('Moved to current X tile...')

    # register to user selected tile to ensure alignment
    def registerToSelectedTile(self, current_path, coordinate_id, coordiante_name, i, j, localPath):

        # old position
        oldX = self.stage.get_pos().x_mm
        oldY = self.stage.get_pos().y_mm
        oldZ = self.stage.get_pos().z_mm

        # input('Adjust microscope manually. and click enter to continue...')
        self.promptUserManualControl()

        # new position set by user
        newX = self.stage.get_pos().x_mm
        newY = self.stage.get_pos().y_mm
        newZ = self.stage.get_pos().z_mm
        # offset all coords by this mount (if multipoint)
        offsetX = newX - oldX
        offsetY = newY - oldY
        offsetZ = newZ - oldZ
        # apply this offset to all scan coordinates
        self.scan_coordinates_mm[:,0] = self.scan_coordinates_mm[:,0] + offsetX
        self.scan_coordinates_mm[:,1] = self.scan_coordinates_mm[:,1] + offsetY
        if len(self.scan_coordinates_mm[0]) == 3:
            self.scan_coordinates_mm[:,2] = self.scan_coordinates_mm[:,2] + offsetZ

    # Ivan registration code
    def Image_registration(self, gray1, gray2):
    
        # gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
    
        # Find keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
    
        # Find matches
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    
        # Extract matched keypoints
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
        # Find homography matrix
        homography, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 10)
    
        # Apply homography to warp the images
        registered_image_ch1 = cv2.warpPerspective(gray2, homography, (gray1.shape[1], gray1.shape[0]), flags = cv2.INTER_LANCZOS4)
        # registered_image_ch2 = cv2.warpPerspective(CH2, homography, (image1.shape[1], image1.shape[0]), flags = cv2.INTER_LANCZOS4)
        # registered_image_ch3 = cv2.warpPerspective(CH3, homography, (image1.shape[1], image1.shape[0]), flags = cv2.INTER_LANCZOS4)
        # registered_image_ch4 = cv2.warpPerspective(CH4, homography, (image1.shape[1], image1.shape[0]), flags = cv2.INTER_LANCZOS4)
    
        # registered_image_ch1_gray = cv2.cvtColor(registered_image_ch1, cv2.COLOR_BGR2GRAY)
        # registered_image_ch2_gray = cv2.cvtColor(registered_image_ch2, cv2.COLOR_BGR2GRAY)
        # registered_image_ch3_gray = cv2.cvtColor(registered_image_ch3, cv2.COLOR_BGR2GRAY)
        # registered_image_ch4_gray = cv2.cvtColor(registered_image_ch4, cv2.COLOR_BGR2GRAY)
    
        return registered_image_ch1, homography

    # compute mm to usteps
    def convert_mm_to_usteps(self, dist):
        mm_per_ustep_X = SCREW_PITCH_X_MM/(self.stage.x_microstepping*FULLSTEPS_PER_REV_X)
        usteps = round(dist/mm_per_ustep_X)
        return usteps

    # check low disk space
    def checkLowDiskSpace(self, current_path):
        # check disk space
        while True:
            total, _, free = shutil.disk_usage(r'/home/cephla/Downloads/')
            if free / total < 0.05: # less than 5% free space
                print('Low disk space warning. Move files to network drive. Acquisition is paused...')
                # move files to network drive
                self.copyLocalFilesToNetwork(self.localPath, current_path)

            else: # enough space
                break        

    # main FOV control function
    async def multipoint_custom_script_entry(self,time_point,current_path,coordinate_id,coordinate_name,i,j, localPath):

        ############################################## User Directories and Inputs ##############################################
        self.use_piezo = True # piezo controller is better

        # stepper motor steps? roughly 2 um if acquisition z step is 0.5 um
        focusStep = 2 if self.use_piezo else 4 * self.deltaZ_usteps # um or steps

        ############################################## END USER INPUTS ##############################################
        # home piezo
        initialPiezoOffset = 0
        if self.use_piezo and self.z_piezo_um != OBJECTIVE_PIEZO_HOME_UM:
            # account for offset to home position
            initialPiezoOffset = self.z_piezo_um - OBJECTIVE_PIEZO_HOME_UM
            self.movePiezo(OBJECTIVE_PIEZO_HOME_UM)
            # offset the actual z coordinate
            self.stage.move_z(initialPiezoOffset/1000) # um to mm

        # check spinning disk is still running
        if self.microscope.xlight.get_disk_motor_state() == 0:
            self.microscope.xlight.set_disk_motor_state(1)
            print('Spinning disk turned back ON')

        # print('focusStep', focusStep)
        # # confocal or widefield mode
        # confocalMode = True
        zCoordsCentered = True # z stack defined symmetrically around center

        # get current XYZ coords for this FOV
        coordinate_mm = self.scan_coordinates_mm[coordinate_id] # mm

        # move to current Z coord minus half autofocus range i.e. bottom of autofocus range
        if self.fillBoundary: # interpolate Z from best fit plane

            # find XY coords of current tile
            x = self.stage.get_pos().x_mm # current X coord
            y = self.stage.get_pos().y_mm # current Y coord
            a, b, c = self.zPlaneBestFit # Coefs a, b, c
            z = a*x + b*y + c # best fit plane equation
            print(f'Moving to interpolated Z coord {z} minus half of autofocus range {focusStep * self.focusCount / 2000} mm...')
            self.stage.move_z_to(z - focusStep * self.focusCount / 2000 + self.zOffset / 1000) # um to mm
            # print(f'Moving to interpolated Z coord {z} mm...')
            # self.stage.move_z_to(z) # um to mm

        else: # use saved Z coord
            # print(f'Moving to Z coord {coordinate_mm[2]} minus one third of autofocus range {focusStep * self.focusCount / 3000} mm...')
            # self.stage.move_z_to(coordinate_mm[2] - focusStep * self.focusCount / 3000) # um to mm
            print(f'Moving to Z coord {coordinate_mm[2]} minus half of autofocus range {focusStep * self.focusCount / 2000} mm...')
            self.stage.move_z_to(coordinate_mm[2] - focusStep * self.focusCount / 2000) # um to mm
            # print(f'Moving to Z coord {coordinate_mm[2]} mm...')
            # self.stage.move_z_to(coordinate_mm[2]) # um to mm
        
        time.sleep(SCAN_STABILIZATION_TIME_MS_Z/1000)
        
        print(f'Time {time_point}, Coord {coordinate_id}, Y {i:03}, X {j:03}')

        # autofocus
        # acquire large DAPI for autofocus. compute autofocus Z coord
        print('Running coarse autofocus scan')
        focusPlane, _ = self.acquireAutofocusScan(focusCount = self.focusCount)
        # fine autofocus scan
        print('Running fine autofocus scan')
        fineFocusCount = 20
        # move to half of z
        self.movePiezo(self.z_piezo_um - fineFocusCount * self.deltaZ / 2) # um
        focusPlane, _ = self.acquireAutofocusScan(focusCount = fineFocusCount, focusStep = self.deltaZ)

        # adjust Z coord and acquire channels at this updated Z coord
        print('Running channel acquisition')
        # run async to check for crashing
        # save to local path in case of network disconnect
        # localPath = Path(r'/home/cephla/Downloads') / Path(*Path(current_path).parts[-4:])
        # localPath.mkdir(parents = True, exist_ok = True)
        task = asyncio.create_task(self.acquireSingleFov(coordinate_name, i, j, time_point, localPath, zCoordsCentered, focusPlane))
        print('Checking for all channel files in reasonable time ...')
        tileFinished = False
        numExpectFiles = len(self.multiPointController.selected_configurations) * self.NZ
        # tile_ID = coordinate_name + str(i) + '_' + str(j if self.x_scan_direction==1 else self.NX-1-j) + '_'
        tile_ID = f'{coordinate_name}_{i}_{j if self.x_scan_direction==1 else self.NX-1-j}_'
        timeStart = time.time()
        while not tileFinished: # check for all channel files

            tifFiles = [f for f in Path(localPath).glob('*.tiff') if f.stem.startswith(tile_ID)] # all tiff files for this tile
            if len(tifFiles) == numExpectFiles:
                tileFinished = True # break loop
                print(f'All channel files found. Tile Time {time_point}, Coord {coordinate_id}, Y {i:03}, X {j:03} is finished')

            if time.time() - timeStart > 15 * 60 and task.done() == False: # 15 min timeout. Script froze
                # run async again. Restart channel acquisition
                print('Script froze. Running channel acquisition again...')
                task.cancel() # cancel previous call
                task = asyncio.create_task(self.acquireSingleFov(coordinate_name, i, j, time_point, localPath, zCoordsCentered, focusPlane))
                timeStart = time.time()

            await asyncio.sleep(2) # wait

        # # get all local files to be copied to network later
        # tile_ID = coordinate_name + str(i) + '_' + str(j if self.x_scan_direction==1 else self.NX-1-j) + '_'
        # tifFiles = [f for f in Path(localPath).glob('*.tiff') if f.stem.startswith(tile_ID)] # all tiff files for this tile
        # tifFiles.append(Path(localPath) / 'log.txt') # add log file file set

        # retract Z axis to preserve oil volume b/t completed tile scans
        if self.zRetract < 0 and j == self.NX - 1 and i == self.NY - 1: # last FOV of current coordinate
            print(f'Retracting Z axis {self.zRetract} um...')
            self.moveZRelative(self.zRetract) # retract Z axis

        # turn off spinning disk if acquisition is finished
        if coordinate_id == self.scan_coordinates_mm.shape[0] - 1 and \
            j == self.NX - 1 and i == self.NY - 1: # last FOV
            print('Last FOV acquired. Turning OFF spinning disk...')
            self.microscope.xlight.set_disk_motor_state(0)

        return tifFiles
    
    async def copyLocalFilesToNetwork(self, localPath, networkPath):
        print('Moving files from local to network...')
        localFiles = [f for f in Path(localPath).glob('*')]
        # Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)\
        # (delayed(self.moveSingleFile)(fileName, networkPath) for fileName in localFiles)
        for _, fileName in enumerate(tqdm(localFiles)):
            # attempt file move
            dest = Path(networkPath) / fileName.name
            try:
                shutil.move(fileName, dest)
            except:
                print('Failed to move', fileName.name)

    def moveSingleFile(self, fileName, networkPath):
        # attempt file move
        dest = Path(networkPath) / fileName.name
        try:
            shutil.move(fileName, dest)
        except:
            print('Failed to move', fileName.name)
        return None
    
    def movePiezo(self, delta_um):
        # print('Moving piezo to', delta_um)
        self.z_piezo_um = delta_um # absolute move
        # dac = int(65535 * (self.z_piezo_um / OBJECTIVE_PIEZO_RANGE_UM))
        # self.stage.microcontroller.analog_write_onboard_DAC(7, dac)
        self.microcontroller.set_piezo_um(self.z_piezo_um)
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE: # for hardware trigger, delay is in waiting for the last row to start exposure
            time.sleep(MULTIPOINT_PIEZO_DELAY_MS/1000)
        if MULTIPOINT_PIEZO_UPDATE_DISPLAY:
            # print('Updating piezo UI to', self.z_piezo_um)
            self.signal_z_piezo_um.emit(self.z_piezo_um)

    # move Z axis relative distance
    def moveZRelative(self, zOffset): # um

        # move Z axis relative distance
        self.stage.move_z(zOffset / 1000) # drop 300 um
        
        time.sleep(SCAN_STABILIZATION_TIME_MS_Z/1000)
        # # maneuver for achiving uniform step size and repeatability when using open-loop control
        # self.stage.move_z_usteps(-160)
        
        # self.stage.move_z_usteps(160)
        
        # time.sleep(SCAN_STABILIZATION_TIME_MS_Z/1000)

        return None

    # compute laplacian for each z plane in parallel
    def computeLapVar(self, plane):
        
        var = cv2.Laplacian(plane, cv2.CV_64F, ksize = 31)
        var = np.var(var)
        
        return var

    # find focus plane via Laplacian variance
    def findFocusLapVar(self, subStack):
        
        # lapVar = Parallel(n_jobs = -1, prefer = 'threads', verbose = 0)\
        # (delayed(computeLapVar)(subStack[ii, :, :].compute()) for ii in range(subStack.shape[0]))

        # use dask instead
        lazy_lap_var = dask.delayed(self.computeLapVar)
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
        
        # # plot gradient and peaks
        # fig, ax = plt.subplots(dpi = 300)
        # ax.scatter(xRange, lapVar, label = 'Variance')
        # ax.scatter(xRange, grad, label = 'Gradient')
        # ax.axhline(mean, label = 'Gradient Mean')
        # ax.scatter(xRange[peaks], grad[peaks], label = 'Peaks')
        # ax.legend()
        
        # find peaks that are more consecutive. 
        # Score peaks based on if adjacent to other peaks
        # peak with consecutive neighbors is focus plane
        
        # idxFocus = np.argmax(grad) + 2
        if len(peaks) == 0: # no peaks found
            print('No peaks found...')
            idxFocus = len(lapVar) // 2 # middle
            
        else:
            idxFocus = peaks[0] + 1 # tallest peak    
            
        if idxFocus > len(lapVar) - 2: # exceeds length, out of bounds
            idxFocus = len(lapVar) - 2
            
        return idxFocus, peaks
    
    # save single plane
    def saveSinglePlane(self, fileOut, image, config, current_path, file_ID):
        
        if image.dtype == np.uint16:

            saving_path = fileOut

            if self.camera.is_color:
                if 'BF LED matrix' in config.name:
                    # if MULTIPOINT_BF_SAVING_OPTION == 'Raw':
                    #     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    if MULTIPOINT_BF_SAVING_OPTION == 'RGB2GRAY':
                        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                    elif MULTIPOINT_BF_SAVING_OPTION == 'Green Channel Only':
                        image = image[:,:,1]

            # do not save if file is empty
            if saving_path == '':
                QApplication.processEvents()
                return image
            
            # # do not overwrite files
            # assert not os.path.exists(saving_path), f'File {saving_path} already exists. Cannot overwrite.'
            
            # keep saving until successful
            success = False
            while not success:
                try:
                    tf.imwrite(saving_path,image)
                    self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                                  line=f'Attempted to save image to {saving_path}')
                except:
                    print('Failed to save image. Retrying...')
                    time.sleep(1)
                else:
                    success = True
                    self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                                  line=f'Successful image save to {saving_path}')
        else:
            saving_path = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '.' + Acquisition.IMAGE_FORMAT)
            # # do not overwrite files
            # assert not os.path.exists(saving_path), f'File {saving_path} already exists. Cannot overwrite.'
            
            if self.camera.is_color:
                if 'BF LED matrix' in config.name:

                    if MULTIPOINT_BF_SAVING_OPTION == 'Raw':
                        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    elif MULTIPOINT_BF_SAVING_OPTION == 'RGB2GRAY':
                        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                    elif MULTIPOINT_BF_SAVING_OPTION == 'Green Channel Only':
                        image = image[:,:,1]
                else:
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            # keep saving until successful
            success = False
            while not success:
                try:
                    cv2.imwrite(saving_path,image)
                except:
                    print('Failed to save image. Retrying...')
                    time.sleep(1)
                else:
                    success = True

    # acquire single plane of single channel
    def acquireSinglePlane(self, fileOut, config, current_path, file_ID):

        # update the current configuration
        self.signal_current_configuration.emit(config)
        
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Changed current configuration to {config.name}')
        # trigger acquisition (including turning on the illumination)
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_on_illumination()
            
            self.camera.send_trigger()
            self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Turned on illumination and sent trigger')
        elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
            self.microcontroller.send_hardware_trigger(control_illumination=True,illumination_on_time_us=self.camera.exposure_time*1000)

        # read camera frame. 
        success = False
        while not success:
            try:
                image = self.camera.read_frame(txtFilePath=Path(current_path,'log.txt'))
            except:
                image = self.camera.read_frame()

            if image is None: # fail
                print('multiPointWorker.camera.read_frame() returned None')
                self.writeLog(txtFilePath=Path(current_path,'log.txt'),
                              line  = 'multiPointWorker.camera.read_frame() returned None')
                time.sleep(2)

                # # reset camera
                # self.camera.stop_streaming()
                # self.camera.close()
                # print('Resetting camera')
                # self.writeLog(txtFilePath=Path(current_path,'log.txt'), line=f'Resetting camera')
                # self.camera = camera.Camera()
                # self.camera.open()
                # self.camera.set_software_triggered_acquisition()
                # # self.camera.set_callback(self.streamHandler.on_new_frame)
                # self.camera.enable_callback()
                # self.camera.start_streaming()
                # print('Finished resetting camera')
                # self.writeLog(txtFilePath=Path(current_path,'log.txt'), line=f'Finished resetting camera')
                # self.camera.start_streaming()

                # # temp fix by turning Live on and off.
                # self.liveController.turn_on_illumination()
                # self.liveController.start_live()
                # 
                # print('Turned on live view')
                # self.writeLog(txtFilePath=Path(current_path,'log.txt'), line=f'Turned on live view')
                # self.liveController.stop_live()
                # self.liveController.turn_off_illumination()
                # 
                # print('Turned off live view')
                # self.writeLog(txtFilePath=Path(current_path,'log.txt'), line=f'Turned off live view')
                
            else: # success
                success = True
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                        line=f'Successfully read camera frame')
                
        # tunr of the illumination if using software trigger
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_off_illumination()
            self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Turned off illumination')
        # process the image -  @@@ to move to camera
        image = utils.crop_image(image,self.crop_width,self.crop_height)
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Cropped image')
        image = utils.rotate_and_flip_image(image,rotate_image_angle=self.camera.rotate_image_angle,flip_image=self.camera.flip_image)
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Rotated and flipped image')
        # self.image_to_display.emit(cv2.resize(image,(round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling)),cv2.INTER_LINEAR))
        image_to_display = utils.crop_image(image,round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling))
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Formatted image for display')
        self.image_to_display.emit(image_to_display)
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Sent display image to display')
        self.image_to_display_multi.emit(image_to_display,config.illumination_source)
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Emit config illumination source')
        self.saveSinglePlane(fileOut, image, config, current_path, file_ID)
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Successful image save')

        QApplication.processEvents()

        return image
    
    def acquire_camera_image(self, config):
     
        # Cephla code
        # update the current configuration
        self.signal_current_configuration.emit(config)
        self.wait_till_operation_is_completed()
        # trigger acquisition (including turning on the illumination) and read frame
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_on_illumination()
            self.wait_till_operation_is_completed()
            self.camera.send_trigger()
            image = self.camera.read_frame()
        elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
            if "Fluorescence" in config.name and ENABLE_NL5 and NL5_USE_DOUT:
                self.camera.image_is_ready = False  # to remove
                self.microscope.nl5.start_acquisition()
                image = self.camera.read_frame(reset_image_ready_flag=False)
            else:
                self.microcontroller.send_hardware_trigger(
                    control_illumination=True, illumination_on_time_us=self.camera.exposure_time * 1000
                )
                image = self.camera.read_frame()
        else:  # continuous acquisition
            image = self.camera.read_frame()

        if image is None:
            print('self.camera.read_frame() returned None')
            self.cameraIssueReconnect()
            sys.exit('Exiting...')

        # turn off the illumination if using software trigger
        if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
            self.liveController.turn_off_illumination()

        # process the image -  @@@ to move to camera
        image = utils.crop_image(image,self.crop_width,self.crop_height)
        image = utils.rotate_and_flip_image(image,rotate_image_angle=self.camera.rotate_image_angle,flip_image=self.camera.flip_image)
        image_to_display = utils.crop_image(image,round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling))
        self.image_to_display.emit(image_to_display)
        self.image_to_display_multi.emit(image_to_display,config.illumination_source)

        # self.save_image(image, file_ID, config, current_path)
        # self.update_napari(image, config.name, i, j, k)

        # current_round_images[config.name] = np.copy(image)

        # self.handle_dpc_generation(current_round_images)
        # self.handle_rgb_generation(current_round_images, file_ID, current_path, i, j, k)

        QApplication.processEvents()

        return image
    
    # error: camera returning None
    def cameraIssueReconnect(self):

        # # reset camera
        # print('Resetting camera')
        # self.camera.stop_streaming()
        # self.camera.close()
        # self.camera = self.multiPointController.camera
        # self.camera.open()
        # self.camera.set_software_triggered_acquisition()
        # # self.camera.set_callback(self.streamHandler.on_new_frame)
        # self.camera.enable_callback()
        # self.camera.start_streaming()
        # print('Finished resetting camera')
        # self.camera.start_streaming()

        # temp fix by turning Live on and off.
        self.liveController.turn_on_illumination()
        self.liveController.start_live()           
        print('Turned on live view')
        # self.liveController.stop_live()
        # self.liveController.turn_off_illumination()
        # print('Turned off live view')
    
    # log file
    def writeLog(self, txtFilePath, line):
        success = False
        while not success:
            try:
                f = open(txtFilePath, 'a')
                f.write('\n' + str(datetime.now()))
                f.write(' ' + line)
                f.close()
            except:
                print('Failed to write log file. Retrying...')
                time.sleep(2)
            else:
                success = True
    
    # skip blank tile
    def acquireBlankTile(self, coordinate_name, i, j, time_point, current_path, zCoordsCentered):

        # iterate through selected modes
        for config in self.selected_configurations:

            # z-stack
            for k in trange(self.NZ): # each z plane
                
                # file_ID = coordinate_name + str(i) + '_' + str(j if self.x_scan_direction==1 else self.NX-1-j) + '_' + str(k)
                file_ID = f'{coordinate_name}_{i}_{j if self.x_scan_direction==1 else self.NX-1-j}_{k}'
                # metadata = dict(x = self.stage.get_pos().x_mm, y = self.stage.get_pos().y_mm, z = self.stage.get_pos().z_mm)
                # metadata = json.dumps(metadata)
                if 'USB Spectrometer' not in config.name:

                    # if time_point%10 != 0:

                    #     if 'Fluorescence' in config.name:
                    #         # only do fluorescence every 10th timepoint
                    #         continue

                    # acquire single plane or multiple if SRRF
                    if self.numFrames > 1: # SRRF acquisition
                        
                        for kk in range(self.numFrames): # acquire repeated frames

                            # print('Acquiring frame', kk + 1, ' / ', numFrames)
                            fileOut = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '_Frame' + str(kk).zfill(3) + '.tiff')
                            if kk == 0:
                                image = self.acquire_camera_image(config)

                            self.saveSinglePlane(fileOut, image, config, current_path, file_ID)

                    else: # normal acquisition

                        fileOut = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '.tiff')
                        if k == 0:
                            image = self.acquire_camera_image(config)
                        self.saveSinglePlane(fileOut, image, config, current_path, file_ID)
                
                else:

                    if self.usb_spectrometer != None:
                        for l in range(N_SPECTRUM_PER_POINT):
                            data = self.usb_spectrometer.read_spectrum()
                            self.spectrum_to_display.emit(data)
                            saving_path = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '_' + str(l) + '.csv')
                            np.savetxt(saving_path,data,delimiter=',')

            # add the coordinate of the current location
            if self.use_piezo:
                new_row = pd.DataFrame({'i':[i],'j':[self.NX-1-j],'k':[k],
                                    'x (mm)':[self.stage.get_pos().x_mm],
                                    'y (mm)':[self.stage.get_pos().y_mm],
                                    'z (um)':[self.stage.get_pos().z_mm*1000],
                                    'z_piezo (um)':[self.z_piezo_um]})
            else:
                new_row = pd.DataFrame({'i':[i],'j':[self.NX-1-j],'k':[k],
                                        'x (mm)':[self.stage.get_pos().x_mm],
                                        'y (mm)':[self.stage.get_pos().y_mm],
                                        'z (um)':[self.stage.get_pos().z_mm*1000]})
            self.coordinates_pd = pd.concat([self.coordinates_pd, new_row], ignore_index=True)

            # register the current fov in the navigationViewer 
            self.signal_register_current_fov.emit(self.stage.get_pos().x_mm,self.stage.get_pos().y_mm)

            # check if the acquisition should be aborted
            if self.multiPointController.abort_acqusition_requested:
                self.liveController.turn_off_illumination()
                # self.stage.move_x_usteps(-self.dx_usteps)
                self.stage.move_x(-self.dx)
                
                # self.stage.move_y_usteps(-self.dy_usteps)
                self.stage.move_y(-self.dy)
                
                # if self.stage.get_pid_control_flag(2) is False:
                #     _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                #     self.stage.move_z_usteps(-self.dz_usteps-_usteps_to_clear_backlash)
                    
                #     self.stage.move_z_usteps(_usteps_to_clear_backlash)
                    
                # else:
                #     self.stage.move_z_usteps(-self.dz_usteps)
                    

                self.coordinates_pd.to_csv(os.path.join(current_path,'coordinates.csv'),index=False,header=True)
                # self.stage.enable_joystick_button_action = True
                return

        # z stack is finished now
        if self.NZ > 1:
            # move z back
            if self.use_piezo:
                self.movePiezo(OBJECTIVE_PIEZO_HOME_UM) # reset piezo
            else:
                if zCoordsCentered == True:
                    if self.stage.get_pid_control_flag(2) is False:
                        _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                        self.stage.move_z_usteps( -self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2) - _usteps_to_clear_backlash)
                        
                        self.stage.move_z_usteps(_usteps_to_clear_backlash)
                        
                    else:
                        self.stage.move_z_usteps( -self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2) )
                        

                    self.dz_usteps = self.dz_usteps - self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2)
                else:
                    if self.stage.get_pid_control_flag(2) is False:
                        _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                        self.stage.move_z_usteps(-self.deltaZ_usteps*(self.NZ-1) - _usteps_to_clear_backlash)
                        
                        self.stage.move_z_usteps(_usteps_to_clear_backlash)
                        
                    else:
                        self.stage.move_z_usteps(-self.deltaZ_usteps*(self.NZ-1))
                        

                    self.dz_usteps = self.dz_usteps - self.deltaZ_usteps*(self.NZ-1)

        # update FOV counter
        self.FOV_counter = self.FOV_counter + 1

    # acquire ZYXC images for single FOV
    async def acquireSingleFov(self, coordinate_name, i, j, time_point, current_path, zCoordsCentered, focusPlane):

        # move pizeo to focus plane
        self.movePiezo(focusPlane) # um
        
        # if tile skipped, acquire single plane and repeat saving
        if self.multiPointController.skipBlank and self.skipTile: # skip
            print('Skipping blank tile', i, j)
            self.acquireBlankTile(coordinate_name, i, j, time_point, current_path, zCoordsCentered)
            return None

        if self.NZ > 1:
            # use piezo to move to bottom of z stack
            self.movePiezo(self.z_piezo_um - self.deltaZ * self.NZ / 2) # um
            startPos = self.z_piezo_um

        # z-stack
        for k in trange(self.NZ): # each z plane
            
            # file_ID = coordinate_name + str(i) + '_' + str(j if self.x_scan_direction==1 else self.NX-1-j) + '_' + str(k)
            file_ID = f'{coordinate_name}_{i}_{j if self.x_scan_direction==1 else self.NX-1-j}_{k}'
            self.writeLog(txtFilePath=Path(current_path, 'log.txt'), 
                          line = f'Acquiring z plane {k} / {self.NZ} for {file_ID}')
            # metadata = dict(x = self.stage.get_pos().x_mm, y = self.stage.get_pos().y_mm, z = self.stage.get_pos().z_mm)
            # metadata = json.dumps(metadata)

            # # print current z plane
            # print() # create spacing so easier to read
            # print('Acquiring z plane', k + 1, ' / ', self.NZ)
            # print()

            # iterate through selected modes
            for config in self.selected_configurations:
                self.writeLog(txtFilePath=Path(current_path, 'log.txt'),
                                line = f'Acquiring {config.name}')

                if 'USB Spectrometer' not in config.name:

                    # if time_point%10 != 0:

                    #     if 'Fluorescence' in config.name:
                    #         # only do fluorescence every 10th timepoint
                    #         continue

                    # acquire single plane or multiple if SRRF
                    if self.numFrames > 1: # SRRF acquisition
                        
                        for kk in range(self.numFrames): # acquire repeated frames

                            # print('Acquiring frame', kk + 1, ' / ', numFrames)

                            fileOut = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '_Frame' + str(kk).zfill(3) + '.tiff')
                            image = self.acquire_camera_image(config)
                            self.saveSinglePlane(fileOut, image, config, current_path, file_ID)

                    else: # normal acquisition

                        fileOut = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '.tiff')
                        image = self.acquire_camera_image(config)
                        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                                          line=f'Acquired camera image for {fileOut}')
                        self.saveSinglePlane(fileOut, image, config, current_path, file_ID)
                        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                                      line=f'Saved single plane for {fileOut}')
                
                else:
                    if self.usb_spectrometer != None:
                        for l in range(N_SPECTRUM_PER_POINT):
                            data = self.usb_spectrometer.read_spectrum()
                            self.spectrum_to_display.emit(data)
                            saving_path = os.path.join(current_path, file_ID + '_' + str(config.name).replace(' ','_') + '_' + str(l) + '.csv')
                            np.savetxt(saving_path,data,delimiter=',')

            # add the coordinate of the current location
            if self.use_piezo:
                new_row = pd.DataFrame({'i':[i],'j':[self.NX-1-j],'k':[k],
                                    'x (mm)':[self.stage.get_pos().x_mm],
                                    'y (mm)':[self.stage.get_pos().y_mm],
                                    'z (um)':[self.stage.get_pos().z_mm*1000],
                                    'z_piezo (um)':[self.z_piezo_um-startPos]})
                self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                              line=f'Added row to coordinates.csv')
            else:
                new_row = pd.DataFrame({'i':[i],'j':[self.NX-1-j],'k':[k],
                                        'x (mm)':[self.stage.get_pos().x_mm],
                                        'y (mm)':[self.stage.get_pos().y_mm],
                                        'z (um)':[self.stage.get_pos().z_mm*1000]})
            self.coordinates_pd = pd.concat([self.coordinates_pd, new_row], ignore_index=True)

            # register the current fov in the navigationViewer 
            self.signal_register_current_fov.emit(self.stage.get_pos().x_mm,self.stage.get_pos().y_mm)
            self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                          line=f'Registered current FOV in navigationViewer')

            # check if the acquisition should be aborted
            if self.multiPointController.abort_acqusition_requested:
                self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                              line=f'Abort acquisition requested')
                self.liveController.turn_off_illumination()
                # self.stage.move_x_usteps(-self.dx_usteps)
                self.stage.move_x(-self.dx)
                
                # self.stage.move_y_usteps(-self.dy_usteps)
                self.stage.move_y(-self.dy)
                
                # if self.stage.get_pid_control_flag(2) is False:
                #     _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                #     self.stage.move_z_usteps(-self.dz_usteps-_usteps_to_clear_backlash)
                    
                #     self.stage.move_z_usteps(_usteps_to_clear_backlash)
                    
                # else:
                #     self.stage.move_z_usteps(-self.dz_usteps)
                    

                self.coordinates_pd.to_csv(os.path.join(current_path,'coordinates.csv'),index=False,header=True)
                # self.stage.enable_joystick_button_action = True
                return

            if self.NZ > 1: 
                # move z
                if k < self.NZ - 1: # increment Z
                    if self.use_piezo:
                        self.movePiezo(self.z_piezo_um + self.deltaZ)
                        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                                      line=f'Moved piezo to {self.z_piezo_um + self.deltaZ}')
                    else:
                        self.stage.move_z_usteps(self.deltaZ_usteps)
                        
                        time.sleep(SCAN_STABILIZATION_TIME_MS_Z/1000)
                        self.dz_usteps = self.dz_usteps + self.deltaZ_usteps

        # z stack is finished now
        if self.NZ > 1:
            # move z back
            if self.use_piezo:
                self.movePiezo(OBJECTIVE_PIEZO_HOME_UM) # reset piezo
                self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                              line=f'Reset piezo to {OBJECTIVE_PIEZO_HOME_UM}')
            else:
                if zCoordsCentered == True:
                    if self.stage.get_pid_control_flag(2) is False:
                        _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                        self.stage.move_z_usteps( -self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2) - _usteps_to_clear_backlash)
                        
                        self.stage.move_z_usteps(_usteps_to_clear_backlash)
                        
                    else:
                        self.stage.move_z_usteps( -self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2) )
                        

                    self.dz_usteps = self.dz_usteps - self.deltaZ_usteps*(self.NZ-1) + self.deltaZ_usteps*round((self.NZ-1)/2)
                else:
                    if self.stage.get_pid_control_flag(2) is False:
                        _usteps_to_clear_backlash = max(160,20*self.stage.z_microstepping)
                        self.stage.move_z_usteps(-self.deltaZ_usteps*(self.NZ-1) - _usteps_to_clear_backlash)
                        
                        self.stage.move_z_usteps(_usteps_to_clear_backlash)
                        
                    else:
                        self.stage.move_z_usteps(-self.deltaZ_usteps*(self.NZ-1))
                        

                    self.dz_usteps = self.dz_usteps - self.deltaZ_usteps*(self.NZ-1)

        # update FOV counter
        self.FOV_counter = self.FOV_counter + 1
        self.writeLog(txtFilePath=Path(current_path,'log.txt'), 
                      line=f'Incremented FOV counter')
        return None

    # acquire large DAPI scan for autofocus
    def acquireAutofocusScan(self, focusCount, focusStep = 2):

        # select focus channel to use
        config = [config for config in self.selected_configurations if self.focusChan == config.name]
        assert len(config) == 1, 'Focus channel not found in selected configurations'
        config = config[0]

        # z-stack
        zStack = []
        zPos = []
        for k in trange(focusCount): # z plane DAPI scan
            
            # acquire image but do not save
            plane = self.acquire_camera_image(config)

            # build z Stack for to compute focus plane later
            plane = da.from_array(plane)
            zStack.append(plane) # each YX

            if k < focusCount - 1: # increment Z

                # use piezo
                # record current piezo position
                zPos.append(self.z_piezo_um) # um
                self.movePiezo(self.z_piezo_um + focusStep)

        imgStack = da.stack(zStack, axis = 0) # ZYX, autofocus scan

        # now compute autofocus plane
        # compute focus plane via blur detection
        # print()
        print('Computing focus plane...')
        # z  = self.scan_coordinates_mm[coordinate_id, 2] * 1e3 # current z coordinate in um
        # zRange = focusStep * focusCount # total range, um
        # zRange = np.linspace(z - zRange / 2, z + zRange / 2, focusCount)   # um  
        idxFocus, peaks = self.findFocusLapVar(imgStack) # all peaks
        # select closest peaks to prev
        if not isinstance(peaks, int) and len(peaks) > 1: # multiple peaks
            print('Multiple peaks found. Finding nearest peak to middle...')
            diff = peaks - focusCount // 2
            diff = np.abs(diff)
            closest = np.argmin(diff)
            idxFocus = peaks[closest]

        print('Idx of focus plane is', idxFocus + 1, ' / ', focusCount) # zero indexed

        # update piezo with focus coordinate
        self.movePiezo(zPos[idxFocus]) # um

        self.checkIfTileBlank(imgStack) # check if blank to skip

        # if autofocus is at extremes of range, increase autofocus range by 10%
        if not self.skipTile and focusCount == self.focusCount: # not blank and using full autofocus range
            lower = 0.2 * self.focusCount
            upper = 0.8 * self.focusCount
            if idxFocus < lower or idxFocus > upper: # at extremes
                newRange = int(self.focusCount * 1.1) # increase by 10%
                # check if new range exceeds 140 limit
                if newRange > 140:
                    newRange = 140
                self.focusCount = newRange # expand range

        return zPos[idxFocus], imgStack # um of focus plane
    
    def checkIfTileBlank(self, imgStack):
        # skip if sample is blank
        self.skipTile = False
        if self.multiPointController.skipBlank:
            # check if blank
            blank = self.multiPointController.blankThresh
            sample = self.multiPointController.sampleThresh
            mip = np.max(imgStack, axis = 0).compute() # YX MIP
            mip = np.var(mip) # variance of MIP
            print('Blank thresh', blank, 'Sample thresh', sample)
            vsBlank = np.abs(mip - blank) # variance of blank is smaller than sample
            vsSample = np.abs(mip - sample)
            print(f'Scan vs Blank = {vsBlank}')
            print(f'Scan vs Sample = {vsSample}')
            if vsBlank < vsSample: # more similar to blank than sample
                # print('Blank Tile detected. Skipping...')
                self.skipTile = True

    # compare histograms
    def compareHist(self, scan, blank, sample):
        scan = self.computeHist(scan)
        blank = self.computeHist(blank)
        sample = self.computeHist(sample)

        # compute Euclidean distance b/t histograms
        vsBlank = self.computeDist(scan, blank)
        vsSample = self.computeDist(scan, sample)
        
        return vsSample, vsBlank
    
    def computeDist(self, hist1, hist2):
        c1 = 0
        i = 0
        while i<len(hist1) and i<len(hist2): 
            c1+=(hist1[i]-hist2[i])**2
            i+= 1
        c1 = c1**(1 / 2) 
        return c1
    
    def computeHist(self, img):
        # bin width 10
        binEdges = np.arange(0, 65500, 10)
        hist, _ = np.histogram(img, bins = binEdges)
        return hist

    def computeMSE(self, imageA, imageB):
        # Compute the Mean Squared Error between the two images
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

class MultiPointController(QObject):

    acquisitionFinished = Signal()
    image_to_display = Signal(np.ndarray)
    image_to_display_multi = Signal(np.ndarray, int)
    spectrum_to_display = Signal(np.ndarray)
    signal_current_configuration = Signal(Configuration)
    signal_register_current_fov = Signal(float, float)
    detection_stats = Signal(object)
    signal_stitcher = Signal(str)
    napari_rtp_layers_update = Signal(np.ndarray, str)
    napari_layers_init = Signal(int, int, object)
    napari_layers_update = Signal(np.ndarray, float, float, int, str)  # image, x_mm, y_mm, k, channel
    signal_z_piezo_um = Signal(float)
    signal_acquisition_progress = Signal(int, int, int)
    signal_region_progress = Signal(int, int)

    def __init__(
        self,
        camera,
        stage: AbstractStage,
        microcontroller: Microcontroller,
        liveController,
        autofocusController,
        configurationManager,
        usb_spectrometer=None,
        scanCoordinates=None,
        parent=None,
        pump = None, # harvard pump
    ):
        QObject.__init__(self)
        self._log = squid.logging.get_logger(self.__class__.__name__)
        self.camera = camera
        if DO_FLUORESCENCE_RTP:
            self.processingHandler = ProcessingHandler()
        self.stage = stage
        self.microcontroller = microcontroller
        self.liveController = liveController
        self.autofocusController = autofocusController
        self.configurationManager = configurationManager
        self.multiPointWorker: Optional[MultiPointWorker] = None
        self.thread: Optional[QThread] = None
        self.NX = 1
        self.NY = 1
        self.NZ = 1
        self.Nt = 1
        self.deltaX = Acquisition.DX
        self.deltaY = Acquisition.DY
        # TODO(imo): Switch all to consistent mm units
        self.deltaZ = Acquisition.DZ / 1000
        self.deltat = 0
        self.do_autofocus = False
        self.do_reflection_af = False
        self.gen_focus_map = False
        self.focus_map_storage = []
        self.already_using_fmap = False
        self.do_segmentation = False
        self.do_fluorescence_rtp = DO_FLUORESCENCE_RTP
        self.crop_width = Acquisition.CROP_WIDTH
        self.crop_height = Acquisition.CROP_HEIGHT
        self.display_resolution_scaling = Acquisition.IMAGE_DISPLAY_SCALING_FACTOR
        self.counter = 0
        self.experiment_ID = None
        self.base_path = None
        self.use_piezo = False  # MULTIPOINT_USE_PIEZO_FOR_ZSTACKS
        self.selected_configurations = []
        self.usb_spectrometer = usb_spectrometer
        self.scanCoordinates = scanCoordinates
        self.scan_region_names = []
        self.scan_region_coords_mm = []
        self.scan_region_fov_coords_mm = {}
        self.parent = parent
        self.start_time = 0
        self.old_images_per_page = 1
        z_mm_current = self.stage.get_pos().z_mm
        self.z_range = [z_mm_current, z_mm_current + self.deltaZ * (self.NZ - 1)]  # [start_mm, end_mm]

        try:
            if self.parent is not None:
                self.old_images_per_page = self.parent.dataHandler.n_images_per_page
        except:
            pass
        self.z_stacking_config = Z_STACKING_CONFIG

        # calibrate sample and blank areas for skipping
        self.blankThresh = -1
        self.sampleThresh = -1
        self.skipBlank = False
        self.pump = pump # harvard pump

        # NEW PARAMETERS FOR setExpParams
        self.focusCount = 100
        self.zRetract = -5000
        self.objective = 60
        self.initialOffset = 0
        self.numFrames = 1
        self.overlapFrac = 0.15
        self.focusChan = 'Fluorescence 405 nm Ex'
        self.stitchMode = 'MIP'
        self.fillBoundary = False
        self.spreadOil = False
        self.registerToPrevTile = False

    def acquisition_in_progress(self):
        if self.thread and self.thread.isRunning() and self.multiPointWorker:
            return True
        return False

    def set_use_piezo(self, checked):
        print("Use Piezo:", checked)
        self.use_piezo = checked
        if self.multiPointWorker:
            self.multiPointWorker.update_use_piezo(checked)

    def set_z_stacking_config(self, z_stacking_config_index):
        if z_stacking_config_index in Z_STACKING_CONFIG_MAP:
            self.z_stacking_config = Z_STACKING_CONFIG_MAP[z_stacking_config_index]
        print(f"z-stacking configuration set to {self.z_stacking_config}")

    def set_z_range(self, minZ, maxZ):
        self.z_range = [minZ, maxZ]

    def set_NX(self, N):
        self.NX = N

    def set_NY(self, N):
        self.NY = N

    def set_NZ(self, N):
        self.NZ = N

    def set_Nt(self, N):
        self.Nt = N

    def set_deltaX(self, delta):
        self.deltaX = delta

    def set_deltaY(self, delta):
        self.deltaY = delta

    def set_deltaZ(self, delta_um):
        self.deltaZ = delta_um / 1000

    def set_deltat(self, delta):
        self.deltat = delta

    def set_af_flag(self, flag):
        self.do_autofocus = flag

    def set_reflection_af_flag(self, flag):
        self.do_reflection_af = flag

    def set_gen_focus_map_flag(self, flag):
        self.gen_focus_map = flag
        if not flag:
            self.autofocusController.set_focus_map_use(False)

    def set_stitch_tiles_flag(self, flag):
        self.do_stitch_tiles = flag

    def set_segmentation_flag(self, flag):
        self.do_segmentation = flag

    def set_fluorescence_rtp_flag(self, flag):
        self.do_fluorescence_rtp = flag

    def set_focus_map(self, focusMap):
        self.focus_map = focusMap  # None if dont use focusMap

    # NEW SETTER METHODS FOR setExpParams PARAMETERS
    def set_focusCount(self, value):
        self.focusCount = value
    
    def set_zRetract(self, value):
        self.zRetract = value
    
    def set_objective(self, value):
        self.objective = value
    
    def set_initialOffset(self, value):
        self.initialOffset = value
    
    def set_numFrames(self, value):
        self.numFrames = value
    
    def set_overlapFrac(self, value):
        self.overlapFrac = value
    
    def set_focusChan(self, value):
        self.focusChan = value
    
    def set_stitchMode(self, value):
        self.stitchMode = value
    
    def set_fillBoundary(self, value):
        self.fillBoundary = value
    
    def set_spreadOil(self, value):
        self.spreadOil = value
    
    def set_registerToPrevTile(self, value):
        self.registerToPrevTile = value
    
    def set_sample_list(self, sample_list):
        """Set the list of samples for multi-sample acquisition"""
        self.sample_list = sample_list
        print(f"Sample list set with {len(sample_list)} samples")
        # Set parameters from first sample for initial configuration
        if sample_list:
            first_sample = sample_list[0]
            # Set exposure times from first sample
            if 'selected_channels' in first_sample:
                self.set_selected_configurations(first_sample['selected_channels'])
            # Set base path
            if 'base_path' in first_sample:
                self.base_path = first_sample['base_path']

    def set_crop(self, crop_width, crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def set_base_path(self, path):
        self.base_path = path

    def start_new_experiment(self,experiment_ID, createFolder = False): # @@@ to do: change name to prepare_folder_for_new_experiment
        # generate unique experiment ID
        # if createFolder:

        # check if existing experiment ID folder already
        folder = [f for f in Path(self.base_path).glob('*') if f.is_dir() and f.stem.startswith('_')]
        if len(folder) == 0:
            self.experiment_ID = experiment_ID.replace(' ','_') + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        else: # use existing folder
            self.experiment_ID = folder[0].name
        
        Path(self.base_path,self.experiment_ID).mkdir(parents=True, exist_ok=True)
        print('Multipoint controller using folder', self.experiment_ID)
        self.recording_start_time = time.time()
        # create a new folder
        # os.mkdir(os.path.join(self.base_path,self.experiment_ID))
        configManagerThrowaway = ConfigurationManager(self.configurationManager.config_filename)
        # for config in self.selected_configurations:
        #     print("config name: ", config.name, 'exposure', config.exposure_time)
        configManagerThrowaway.write_configuration_selected(self.selected_configurations,os.path.join(self.base_path,self.experiment_ID)+"/configurations.xml") # save the configuration for the experiment

        acquisition_parameters = {'dx(mm)':self.deltaX, 'Nx':self.NX, 'dy(mm)':self.deltaY, 'Ny':self.NY, 'dz(um)':self.deltaZ*1000,'Nz':self.NZ,'dt(s)':self.deltat,'Nt':self.Nt,'with AF':self.do_autofocus,'with reflection AF':self.do_reflection_af}
        try: # write objective data if it is available
            current_objective = self.parent.objectiveStore.current_objective
            objective_info = self.parent.objectiveStore.objectives_dict.get(current_objective, {})
            acquisition_parameters['objective'] = {}
            for k in objective_info.keys():
                acquisition_parameters['objective'][k]=objective_info[k]
            acquisition_parameters['objective']['name']=current_objective
        except:
            try:
                objective_info = OBJECTIVES[DEFAULT_OBJECTIVE]
                acquisition_parameters['objective'] = {}
                for k in objective_info.keys():
                    acquisition_parameters['objective'][k] = objective_info[k]
                acquisition_parameters['objective']['name']=DEFAULT_OBJECTIVE
            except:
                pass
        acquisition_parameters['sensor_pixel_size_um'] = CAMERA_PIXEL_SIZE_UM[CAMERA_SENSOR]
        acquisition_parameters['tube_lens_mm'] = TUBE_LENS_MM
        f = open(os.path.join(self.base_path,self.experiment_ID)+"/acquisition parameters.json","w")
        f.write(json.dumps(acquisition_parameters))
        f.close()

    def set_selected_configurations(self, selected_configurations_name):
        self.selected_configurations = []
        for configuration_name in selected_configurations_name:
            self.selected_configurations.append(
                next(
                    (config for config in self.configurationManager.configurations if config.name == configuration_name)
                )
            )

    def run_acquisition(self, sample_list=None, location_list=None): # @@@ to do: change name to run_experiment

        # if user calibrated blank and sample areas, use skipping
        # if np.var(self.blankThresh > 0) or np.var(self.sampleThresh) > 0:
        if self.blankThresh > 0 or self.sampleThresh > 0:
            # if np.var(self.sampleThresh) <= 0: # assign based on blank
            if self.sampleThresh <= 0:
                self.sampleThresh = self.blankThresh * 5
            self.skipBlank = True

        print('start multipoint')
        
        # Handle sample_list vs location_list
        if sample_list is not None:
            print(f'Sample list with {len(sample_list)} samples')
            self.sample_list = sample_list
            # Use location_list from first sample if available
            if sample_list and 'location_list' in sample_list[0]:
                self.location_list = sample_list[0]['location_list']
                # Also extract location_ids if available
                if 'location_ids' in sample_list[0]:
                    self.location_ids = sample_list[0]['location_ids']
                else:
                    self.location_ids = None
            else:
                self.location_list = None
                self.location_ids = None
        elif location_list is not None:
            print(location_list)
            self.location_list = location_list
            self.location_ids = None  # No location_ids available in legacy mode
            # Create single sample from location_list for backwards compatibility
            self.sample_list = [{
                'location_list': location_list,
                'nx': self.NX,
                'ny': self.NY, 
                'nz': self.NZ,
                'dz': self.deltaZ * 1000,  # Convert back to μm
                'base_path': getattr(self, 'base_path', ''),
                'selected_channels': getattr(self, 'selected_configurations', [])
            }]
        else:
            self.location_list = None
            self.location_ids = None
            self.sample_list = None

        print(str(self.Nt) + '_' + str(self.NX) + '_' + str(self.NY) + '_' + str(self.NZ))

        self.abort_acqusition_requested = False

        self.configuration_before_running_multipoint = self.liveController.currentConfiguration
        # self.configuration_before_running_multipoint = []
        # stop live
        if self.liveController.is_live:
            self.liveController_was_live_before_multipoint = True
            self.liveController.stop_live() # @@@ to do: also uncheck the live button
        else:
            self.liveController_was_live_before_multipoint = False

        # disable callback
        if self.camera.callback_is_enabled:
            self.camera_callback_was_enabled_before_multipoint = True
            self.camera.disable_callback()
        else:
            self.camera_callback_was_enabled_before_multipoint = False

        if self.usb_spectrometer != None:
            if self.usb_spectrometer.streaming_started == True and self.usb_spectrometer.streaming_paused == False:
                self.usb_spectrometer.pause_streaming()
                self.usb_spectrometer_was_streaming = True
            else:
                self.usb_spectrometer_was_streaming = False

        if self.parent is not None:
            try:
                self.parent.imageDisplayTabs.setCurrentWidget(self.parent.imageArrayDisplayWindow.widget)
            except:
                pass
            try:
                self.parent.recordTabWidget.setCurrentWidget(self.parent.statsDisplayWidget)
            except:
                pass
        
        # run the acquisition
        self.timestamp_acquisition_started = time.time()

        if SHOW_TILED_PREVIEW:
            self.navigationController.keep_scan_begin_position(self.navigationController.x_pos_mm, self.navigationController.y_pos_mm)

        # create a QThread object
        if self.gen_focus_map and not self.do_reflection_af:
            print("Generating focus map for multipoint grid")
            starting_x_mm = self.navigationController.x_pos_mm
            starting_y_mm = self.navigationController.y_pos_mm
            fmap_Nx = max(2,self.NX-1)
            fmap_Ny = max(2,self.NY-1)
            fmap_dx = self.deltaX
            fmap_dy = self.deltaY
            if abs(fmap_dx) < 0.1 and fmap_dx != 0.0:
                fmap_dx = 0.1*fmap_dx/(abs(fmap_dx))
            elif fmap_dx == 0.0:
                fmap_dx = 0.1
            if abs(fmap_dy) < 0.1 and fmap_dy != 0.0:
                 fmap_dy = 0.1*fmap_dy/(abs(fmap_dy))
            elif fmap_dy == 0.0:
                fmap_dy = 0.1
            try:
                self.focus_map_storage = []
                self.already_using_fmap = self.autofocusController.use_focus_map
                for x,y,z in self.autofocusController.focus_map_coords:
                    self.focus_map_storage.append((x,y,z))
                coord1 = (starting_x_mm, starting_y_mm)
                coord2 = (starting_x_mm+fmap_Nx*fmap_dx,starting_y_mm)
                coord3 = (starting_x_mm,starting_y_mm+fmap_Ny*fmap_dy)
                self.autofocusController.gen_focus_map(coord1, coord2, coord3)
                self.autofocusController.set_focus_map_use(True)
                self.navigationController.move_to(starting_x_mm, starting_y_mm)
                self.navigationController.microcontroller.wait_till_operation_is_completed()
            except ValueError:
                print("Invalid coordinates for focus map, aborting.")
                return

        self.thread = QThread()
        # create a worker object
        self.processingHandler.start_processing()
        self.processingHandler.start_uploading()
        self.multiPointWorker = MultiPointWorker(self)
        # move the worker to the thread
        self.multiPointWorker.moveToThread(self.thread)
        # connect signals and slots
        self.thread.started.connect(self.multiPointWorker.run)
        self.multiPointWorker.signal_detection_stats.connect(self.slot_detection_stats)
        self.multiPointWorker.finished.connect(self._on_acquisition_completed)
        self.multiPointWorker.finished.connect(self.multiPointWorker.deleteLater)
        self.multiPointWorker.finished.connect(self.thread.quit)
        self.multiPointWorker.image_to_display.connect(self.slot_image_to_display)
        self.multiPointWorker.image_to_display_multi.connect(self.slot_image_to_display_multi)
        # self.multiPointWorker.image_to_display_tiled_preview.connect(self.slot_image_to_display_tiled_preview)
        self.multiPointWorker.spectrum_to_display.connect(self.slot_spectrum_to_display)
        self.multiPointWorker.signal_current_configuration.connect(self.slot_current_configuration,type=Qt.BlockingQueuedConnection)
        self.multiPointWorker.signal_register_current_fov.connect(self.slot_register_current_fov)
        self.multiPointWorker.napari_layers_init.connect(self.slot_napari_layers_init)
        self.multiPointWorker.napari_layers_update.connect(self.slot_napari_layers_update)
        self.multiPointWorker.signal_z_piezo_um.connect(self.slot_z_piezo_um)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.thread.quit)
        # start the thread
        self.thread.start()

    def _on_acquisition_completed(self):
        self._log.debug("MultiPointController._on_acquisition_completed called")
        # restore the previous selected mode
        if self.gen_focus_map:
            self.autofocusController.clear_focus_map()
            for x, y, z in self.focus_map_storage:
                self.autofocusController.focus_map_coords.append((x, y, z))
            self.autofocusController.use_focus_map = self.already_using_fmap
        self.signal_current_configuration.emit(self.configuration_before_running_multipoint)

        # re-enable callback
        if self.camera_callback_was_enabled_before_multipoint:
            self.camera.enable_callback()
            self.camera_callback_was_enabled_before_multipoint = False

        # re-enable live if it's previously on
        if self.liveController_was_live_before_multipoint:
            self.liveController.start_live()

        if self.usb_spectrometer != None:
            if self.usb_spectrometer_was_streaming:
                self.usb_spectrometer.resume_streaming()

        # emit the acquisition finished signal to enable the UI
        if self.parent is not None:
            try:
                # self.parent.dataHandler.set_number_of_images_per_page(self.old_images_per_page)
                self.parent.dataHandler.sort("Sort by prediction score")
                self.parent.dataHandler.signal_populate_page0.emit()
            except:
                pass
        print("total time for acquisition + processing + reset:", time.time() - self.recording_start_time)
        utils.create_done_file(os.path.join(self.base_path, self.experiment_ID))
        self.acquisitionFinished.emit()
        if not self.abort_acqusition_requested:
            self.signal_stitcher.emit(os.path.join(self.base_path, self.experiment_ID))
        QApplication.processEvents()

    def request_abort_aquisition(self):
        self.abort_acqusition_requested = True

    def slot_detection_stats(self, stats):
        self.detection_stats.emit(stats)

    def slot_image_to_display(self, image):
        self.image_to_display.emit(image)

    def slot_spectrum_to_display(self, data):
        self.spectrum_to_display.emit(data)

    def slot_image_to_display_multi(self, image, illumination_source):
        self.image_to_display_multi.emit(image, illumination_source)

    def slot_current_configuration(self, configuration):
        self.signal_current_configuration.emit(configuration)

    def slot_register_current_fov(self, x_mm, y_mm):
        self.signal_register_current_fov.emit(x_mm, y_mm)

    def slot_napari_rtp_layers_update(self, image, channel):
        self.napari_rtp_layers_update.emit(image, channel)

    def slot_napari_layers_init(self, image_height, image_width, dtype):
        self.napari_layers_init.emit(image_height, image_width, dtype)

    def slot_napari_layers_update(self, image, x_mm, y_mm, k, channel):
        self.napari_layers_update.emit(image, x_mm, y_mm, k, channel)

    def slot_z_piezo_um(self, displacement_um):
        self.signal_z_piezo_um.emit(displacement_um)

    def slot_acquisition_progress(self, current_region, total_regions, current_time_point):
        self.signal_acquisition_progress.emit(current_region, total_regions, current_time_point)

    def slot_region_progress(self, current_fov, total_fovs):
        self.signal_region_progress.emit(current_fov, total_fovs)

class TrackingController(QObject):

    signal_tracking_stopped = Signal()
    image_to_display = Signal(np.ndarray)
    image_to_display_multi = Signal(np.ndarray, int)
    signal_current_configuration = Signal(Configuration)

    def __init__(
        self,
        camera,
        microcontroller: Microcontroller,
        stage: AbstractStage,
        configurationManager,
        liveController: LiveController,
        autofocusController,
        imageDisplayWindow,
    ):
        QObject.__init__(self)
        self.camera = camera
        self.microcontroller = microcontroller
        self.stage = stage
        self.configurationManager = configurationManager
        self.liveController = liveController
        self.autofocusController = autofocusController
        self.imageDisplayWindow = imageDisplayWindow
        self.tracker = tracking.Tracker_Image()

        self.tracking_time_interval_s = 0

        self.crop_width = Acquisition.CROP_WIDTH
        self.crop_height = Acquisition.CROP_HEIGHT
        self.display_resolution_scaling = Acquisition.IMAGE_DISPLAY_SCALING_FACTOR
        self.counter = 0
        self.experiment_ID = None
        self.base_path = None
        self.selected_configurations = []

        self.flag_stage_tracking_enabled = True
        self.flag_AF_enabled = False
        self.flag_save_image = False
        self.flag_stop_tracking_requested = False

        self.pixel_size_um = None
        self.objective = None

    def start_tracking(self):

        # save pre-tracking configuration
        print("start tracking")
        self.configuration_before_running_tracking = self.liveController.currentConfiguration

        # stop live
        if self.liveController.is_live:
            self.was_live_before_tracking = True
            self.liveController.stop_live()  # @@@ to do: also uncheck the live button
        else:
            self.was_live_before_tracking = False

        # disable callback
        if self.camera.callback_is_enabled:
            self.camera_callback_was_enabled_before_tracking = True
            self.camera.disable_callback()
        else:
            self.camera_callback_was_enabled_before_tracking = False

        # hide roi selector
        self.imageDisplayWindow.hide_ROI_selector()

        # run tracking
        self.flag_stop_tracking_requested = False
        # create a QThread object
        try:
            if self.thread.isRunning():
                print("*** previous tracking thread is still running ***")
                self.thread.terminate()
                self.thread.wait()
                print("*** previous tracking threaded manually stopped ***")
        except:
            pass
        self.thread = QThread()
        # create a worker object
        self.trackingWorker = TrackingWorker(self)
        # move the worker to the thread
        self.trackingWorker.moveToThread(self.thread)
        # connect signals and slots
        self.thread.started.connect(self.trackingWorker.run)
        self.trackingWorker.finished.connect(self._on_tracking_stopped)
        self.trackingWorker.finished.connect(self.trackingWorker.deleteLater)
        self.trackingWorker.finished.connect(self.thread.quit)
        self.trackingWorker.image_to_display.connect(self.slot_image_to_display)
        self.trackingWorker.image_to_display_multi.connect(self.slot_image_to_display_multi)
        self.trackingWorker.signal_current_configuration.connect(
            self.slot_current_configuration, type=Qt.BlockingQueuedConnection
        )
        # self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.thread.quit)
        # start the thread
        self.thread.start()

    def _on_tracking_stopped(self):

        # restore the previous selected mode
        self.signal_current_configuration.emit(self.configuration_before_running_tracking)

        # re-enable callback
        if self.camera_callback_was_enabled_before_tracking:
            self.camera.enable_callback()
            self.camera_callback_was_enabled_before_tracking = False

        # re-enable live if it's previously on
        if self.was_live_before_tracking:
            self.liveController.start_live()

        # show ROI selector
        self.imageDisplayWindow.show_ROI_selector()

        # emit the acquisition finished signal to enable the UI
        self.signal_tracking_stopped.emit()
        QApplication.processEvents()

    def start_new_experiment(self, experiment_ID):  # @@@ to do: change name to prepare_folder_for_new_experiment
        # generate unique experiment ID
        self.experiment_ID = experiment_ID + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        self.recording_start_time = time.time()
        # create a new folder
        try:
            utils.ensure_directory_exists(os.path.join(self.base_path, self.experiment_ID))
            self.configurationManager.write_configuration(
                os.path.join(self.base_path, self.experiment_ID) + "/configurations.xml"
            )  # save the configuration for the experiment
        except:
            print("error in making a new folder")
            pass

    def set_selected_configurations(self, selected_configurations_name):
        self.selected_configurations = []
        for configuration_name in selected_configurations_name:
            self.selected_configurations.append(
                next(
                    (config for config in self.configurationManager.configurations if config.name == configuration_name)
                )
            )

    def toggle_stage_tracking(self, state):
        self.flag_stage_tracking_enabled = state > 0
        print("set stage tracking enabled to " + str(self.flag_stage_tracking_enabled))

    def toggel_enable_af(self, state):
        self.flag_AF_enabled = state > 0
        print("set af enabled to " + str(self.flag_AF_enabled))

    def toggel_save_images(self, state):
        self.flag_save_image = state > 0
        print("set save images to " + str(self.flag_save_image))

    def set_base_path(self, path):
        self.base_path = path

    def stop_tracking(self):
        self.flag_stop_tracking_requested = True
        print("stop tracking requested")

    def slot_image_to_display(self, image):
        self.image_to_display.emit(image)

    def slot_image_to_display_multi(self, image, illumination_source):
        self.image_to_display_multi.emit(image, illumination_source)

    def slot_current_configuration(self, configuration):
        self.signal_current_configuration.emit(configuration)

    def update_pixel_size(self, pixel_size_um):
        self.pixel_size_um = pixel_size_um

    def update_tracker_selection(self, tracker_str):
        self.tracker.update_tracker_type(tracker_str)

    def set_tracking_time_interval(self, time_interval):
        self.tracking_time_interval_s = time_interval

    def update_image_resizing_factor(self, image_resizing_factor):
        self.image_resizing_factor = image_resizing_factor
        print("update tracking image resizing factor to " + str(self.image_resizing_factor))
        self.pixel_size_um_scaled = self.pixel_size_um / self.image_resizing_factor

    # PID-based tracking
    """
    def on_new_frame(self,image,frame_ID,timestamp):
        # initialize the tracker when a new track is started
        if self.tracking_frame_counter == 0:
            # initialize the tracker
            # initialize the PID controller
            pass

        # crop the image, resize the image
        # [to fill]

        # get the location
        [x,y] = self.tracker_xy.track(image)
        z = self.track_z.track(image)

        # get motion commands
        dx = self.pid_controller_x.get_actuation(x)
        dy = self.pid_controller_y.get_actuation(y)
        dz = self.pid_controller_z.get_actuation(z)

        # read current location from the microcontroller
        current_stage_position = self.microcontroller.read_received_packet()

        # save the coordinate information (possibly enqueue image for saving here to if a separate ImageSaver object is being used) before the next movement
        # [to fill]

        # generate motion commands
        motion_commands = self.generate_motion_commands(self,dx,dy,dz)

        # send motion commands
        self.microcontroller.send_command(motion_commands)

    def start_a_new_track(self):
        self.tracking_frame_counter = 0
    """


class TrackingWorker(QObject):

    finished = Signal()
    image_to_display = Signal(np.ndarray)
    image_to_display_multi = Signal(np.ndarray, int)
    signal_current_configuration = Signal(Configuration)

    def __init__(self, trackingController: TrackingController):
        QObject.__init__(self)
        self.trackingController = trackingController

        self.camera = self.trackingController.camera
        self.stage = self.trackingController.stage
        self.microcontroller = self.trackingController.microcontroller
        self.liveController = self.trackingController.liveController
        self.autofocusController = self.trackingController.autofocusController
        self.configurationManager = self.trackingController.configurationManager
        self.imageDisplayWindow = self.trackingController.imageDisplayWindow
        self.crop_width = self.trackingController.crop_width
        self.crop_height = self.trackingController.crop_height
        self.display_resolution_scaling = self.trackingController.display_resolution_scaling
        self.counter = self.trackingController.counter
        self.experiment_ID = self.trackingController.experiment_ID
        self.base_path = self.trackingController.base_path
        self.selected_configurations = self.trackingController.selected_configurations
        self.tracker = trackingController.tracker

        self.number_of_selected_configurations = len(self.selected_configurations)

        self.image_saver = ImageSaver_Tracking(
            base_path=os.path.join(self.base_path, self.experiment_ID), image_format="bmp"
        )

    def run(self):

        tracking_frame_counter = 0
        t0 = time.time()

        # save metadata
        self.txt_file = open(os.path.join(self.base_path, self.experiment_ID, "metadata.txt"), "w+")
        self.txt_file.write("t0: " + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f") + "\n")
        self.txt_file.write("objective: " + self.trackingController.objective + "\n")
        self.txt_file.close()

        # create a file for logging
        self.csv_file = open(os.path.join(self.base_path, self.experiment_ID, "track.csv"), "w+")
        self.csv_file.write(
            "dt (s), x_stage (mm), y_stage (mm), z_stage (mm), x_image (mm), y_image(mm), image_filename\n"
        )

        # reset tracker
        self.tracker.reset()

        # get the manually selected roi
        init_roi = self.imageDisplayWindow.get_roi_bounding_box()
        self.tracker.set_roi_bbox(init_roi)

        # tracking loop
        while not self.trackingController.flag_stop_tracking_requested:
            print("tracking_frame_counter: " + str(tracking_frame_counter))
            if tracking_frame_counter == 0:
                is_first_frame = True
            else:
                is_first_frame = False

            # timestamp
            timestamp_last_frame = time.time()

            # switch to the tracking config
            config = self.selected_configurations[0]
            self.signal_current_configuration.emit(config)
            self.microcontroller.wait_till_operation_is_completed()
            # do autofocus
            if self.trackingController.flag_AF_enabled and tracking_frame_counter > 1:
                # do autofocus
                print(">>> autofocus")
                self.autofocusController.autofocus()
                self.autofocusController.wait_till_autofocus_has_completed()
                print(">>> autofocus completed")

            # get current position
            pos = self.stage.get_pos()

            # grab an image
            config = self.selected_configurations[0]
            if self.number_of_selected_configurations > 1:
                self.signal_current_configuration.emit(config)
                # TODO(imo): replace with illumination controller
                self.microcontroller.wait_till_operation_is_completed()
                self.liveController.turn_on_illumination()  # keep illumination on for single configuration acqusition
                self.microcontroller.wait_till_operation_is_completed()
            t = time.time()
            self.camera.send_trigger()
            image = self.camera.read_frame()
            if self.number_of_selected_configurations > 1:
                self.liveController.turn_off_illumination()  # keep illumination on for single configuration acqusition
            # image crop, rotation and flip
            image = utils.crop_image(image, self.crop_width, self.crop_height)
            image = np.squeeze(image)
            image = utils.rotate_and_flip_image(image, rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE)
            # get image size
            image_shape = image.shape
            image_center = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])

            # image the rest configurations
            for config_ in self.selected_configurations[1:]:
                self.signal_current_configuration.emit(config_)
                # TODO(imo): replace with illumination controller
                self.microcontroller.wait_till_operation_is_completed()
                self.liveController.turn_on_illumination()
                self.microcontroller.wait_till_operation_is_completed()
                # TODO(imo): this is broken if we are using hardware triggering
                self.camera.send_trigger()
                image_ = self.camera.read_frame()
                # TODO(imo): use illumination controller
                self.liveController.turn_off_illumination()
                image_ = utils.crop_image(image_, self.crop_width, self.crop_height)
                image_ = np.squeeze(image_)
                image_ = utils.rotate_and_flip_image(
                    image_, rotate_image_angle=ROTATE_IMAGE_ANGLE, flip_image=FLIP_IMAGE
                )
                # display image
                image_to_display_ = utils.crop_image(
                    image_,
                    round(self.crop_width * self.liveController.display_resolution_scaling),
                    round(self.crop_height * self.liveController.display_resolution_scaling),
                )
                self.image_to_display_multi.emit(image_to_display_, config_.illumination_source)
                # save image
                if self.trackingController.flag_save_image:
                    if self.camera.is_color:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.image_saver.enqueue(image_, tracking_frame_counter, str(config_.name))

            # track
            object_found, centroid, rect_pts = self.tracker.track(image, None, is_first_frame=is_first_frame)
            if not object_found:
                print("tracker: object not found")
                break
            in_plane_position_error_pixel = image_center - centroid
            in_plane_position_error_mm = (
                in_plane_position_error_pixel * self.trackingController.pixel_size_um_scaled / 1000
            )
            x_error_mm = in_plane_position_error_mm[0]
            y_error_mm = in_plane_position_error_mm[1]

            # display the new bounding box and the image
            self.imageDisplayWindow.update_bounding_box(rect_pts)
            self.imageDisplayWindow.display_image(image)

            # move
            if self.trackingController.flag_stage_tracking_enabled:
                # TODO(imo): This needs testing!
                self.stage.move_x(x_error_mm)
                self.stage.move_y(y_error_mm)

            # save image
            if self.trackingController.flag_save_image:
                self.image_saver.enqueue(image, tracking_frame_counter, str(config.name))

            # save position data
            self.csv_file.write(
                str(t)
                + ","
                + str(pos.x_mm)
                + ","
                + str(pos.y_mm)
                + ","
                + str(pos.z_mm)
                + ","
                + str(x_error_mm)
                + ","
                + str(y_error_mm)
                + ","
                + str(tracking_frame_counter)
                + "\n"
            )
            if tracking_frame_counter % 100 == 0:
                self.csv_file.flush()

            # wait till tracking interval has elapsed
            while time.time() - timestamp_last_frame < self.trackingController.tracking_time_interval_s:
                time.sleep(0.005)

            # increament counter
            tracking_frame_counter = tracking_frame_counter + 1

        # tracking terminated
        self.csv_file.close()
        self.image_saver.close()
        self.finished.emit()


class ImageDisplayWindow(QMainWindow):

    image_click_coordinates = Signal(int, int, int, int)

    def __init__(
        self,
        liveController=None,
        contrastManager=None,
        window_title="",
        draw_crosshairs=False,
        show_LUT=False,
        autoLevels=False,
    ):
        super().__init__()
        self.liveController = liveController
        self.contrastManager = contrastManager
        self.setWindowTitle(window_title)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.widget = QWidget()
        self.show_LUT = show_LUT
        self.autoLevels = autoLevels

        # interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.view = self.graphics_widget.addViewBox()
        self.graphics_widget.view.invertY()

        ## lock the aspect ratio so pixels are always square
        self.graphics_widget.view.setAspectLocked(True)

        ## Create image item
        if self.show_LUT:
            self.graphics_widget.view = pg.ImageView()
            self.graphics_widget.img = self.graphics_widget.view.getImageItem()
            self.graphics_widget.img.setBorder("w")
            self.graphics_widget.view.ui.roiBtn.hide()
            self.graphics_widget.view.ui.menuBtn.hide()
            self.LUTWidget = self.graphics_widget.view.getHistogramWidget()
            self.LUTWidget.region.sigRegionChanged.connect(self.update_contrast_limits)
            self.LUTWidget.region.sigRegionChangeFinished.connect(self.update_contrast_limits)
        else:
            self.graphics_widget.img = pg.ImageItem(border="w")
            self.graphics_widget.view.addItem(self.graphics_widget.img)

        ## Create ROI
        self.roi_pos = (500, 500)
        self.roi_size = (500, 500)
        self.ROI = pg.ROI(self.roi_pos, self.roi_size, scaleSnap=True, translateSnap=True)
        self.ROI.setZValue(10)
        self.ROI.addScaleHandle((0, 0), (1, 1))
        self.ROI.addScaleHandle((1, 1), (0, 0))
        self.graphics_widget.view.addItem(self.ROI)
        self.ROI.hide()
        self.ROI.sigRegionChanged.connect(self.update_ROI)
        self.roi_pos = self.ROI.pos()
        self.roi_size = self.ROI.size()

        ## Variables for annotating images
        self.draw_rectangle = False
        self.ptRect1 = None
        self.ptRect2 = None
        self.DrawCirc = False
        self.centroid = None
        self.DrawCrossHairs = False
        self.image_offset = np.array([0, 0])

        ## Layout
        layout = QGridLayout()
        if self.show_LUT:
            layout.addWidget(self.graphics_widget.view, 0, 0)
        else:
            layout.addWidget(self.graphics_widget, 0, 0)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)

        # set window size
        desktopWidget = QDesktopWidget()
        width = min(desktopWidget.height() * 0.9, 1000)
        height = width
        self.setFixedSize(int(width), int(height))

        # Connect mouse click handler
        if self.show_LUT:
            self.graphics_widget.view.getView().scene().sigMouseClicked.connect(self.handle_mouse_click)
        else:
            self.graphics_widget.view.scene().sigMouseClicked.connect(self.handle_mouse_click)

    def handle_mouse_click(self, evt):
        # Only process double clicks
        if not evt.double():
            return

        try:
            pos = evt.pos()
            if self.show_LUT:
                view_coord = self.graphics_widget.view.getView().mapSceneToView(pos)
            else:
                view_coord = self.graphics_widget.view.mapSceneToView(pos)
            image_coord = self.graphics_widget.img.mapFromView(view_coord)
        except:
            return

        if self.is_within_image(image_coord):
            x_pixel_centered = int(image_coord.x() - self.graphics_widget.img.width() / 2)
            y_pixel_centered = int(image_coord.y() - self.graphics_widget.img.height() / 2)
            self.image_click_coordinates.emit(
                x_pixel_centered, y_pixel_centered, self.graphics_widget.img.width(), self.graphics_widget.img.height()
            )

    def is_within_image(self, coordinates):
        try:
            image_width = self.graphics_widget.img.width()
            image_height = self.graphics_widget.img.height()
            return 0 <= coordinates.x() < image_width and 0 <= coordinates.y() < image_height
        except:
            return False

    # [Rest of the methods remain exactly the same...]
    def display_image(self, image):
        if ENABLE_TRACKING:
            image = np.copy(image)
            self.image_height, self.image_width = image.shape[:2]
            if self.draw_rectangle:
                cv2.rectangle(image, self.ptRect1, self.ptRect2, (255, 255, 255), 4)
                self.draw_rectangle = False

        info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else np.finfo(image.dtype)
        min_val, max_val = info.min, info.max

        if self.liveController is not None and self.contrastManager is not None:
            channel_name = self.liveController.currentConfiguration.name
            if self.contrastManager.acquisition_dtype != None and self.contrastManager.acquisition_dtype != np.dtype(
                image.dtype
            ):
                self.contrastManager.scale_contrast_limits(np.dtype(image.dtype))
            min_val, max_val = self.contrastManager.get_limits(channel_name, image.dtype)

        self.graphics_widget.img.setImage(image, autoLevels=self.autoLevels, levels=(min_val, max_val))

        if not self.autoLevels:
            if self.show_LUT:
                self.LUTWidget.setLevels(min_val, max_val)
                self.LUTWidget.setHistogramRange(info.min, info.max)
            else:
                self.graphics_widget.img.setLevels((min_val, max_val))

        self.graphics_widget.img.updateImage()

    def update_contrast_limits(self):
        if self.show_LUT and self.contrastManager and self.contrastManager.acquisition_dtype:
            min_val, max_val = self.LUTWidget.region.getRegion()
            self.contrastManager.update_limits(self.liveController.currentConfiguration.name, min_val, max_val)

    def update_ROI(self):
        self.roi_pos = self.ROI.pos()
        self.roi_size = self.ROI.size()

    def show_ROI_selector(self):
        self.ROI.show()

    def hide_ROI_selector(self):
        self.ROI.hide()

    def get_roi(self):
        return self.roi_pos, self.roi_size

    def update_bounding_box(self, pts):
        self.draw_rectangle = True
        self.ptRect1 = (pts[0][0], pts[0][1])
        self.ptRect2 = (pts[1][0], pts[1][1])

    def get_roi_bounding_box(self):
        self.update_ROI()
        width = self.roi_size[0]
        height = self.roi_size[1]
        xmin = max(0, self.roi_pos[0])
        ymin = max(0, self.roi_pos[1])
        return np.array([xmin, ymin, width, height])

    def set_autolevel(self, enabled):
        self.autoLevels = enabled
        print("set autolevel to " + str(enabled))


class NavigationViewer(QFrame):

    signal_coordinates_clicked = Signal(float, float)  # Will emit x_mm, y_mm when clicked

    def __init__(self, objectivestore, sample="glass slide", invertX=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = squid.logging.get_logger(self.__class__.__name__)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.sample = sample
        self.objectiveStore = objectivestore
        self.well_size_mm = WELL_SIZE_MM
        self.well_spacing_mm = WELL_SPACING_MM
        self.number_of_skip = NUMBER_OF_SKIP
        self.a1_x_mm = A1_X_MM
        self.a1_y_mm = A1_Y_MM
        self.a1_x_pixel = A1_X_PIXEL
        self.a1_y_pixel = A1_Y_PIXEL
        self.location_update_threshold_mm = 0.2
        self.box_color = (255, 0, 0)
        self.box_line_thickness = 2
        self.acquisition_size = Acquisition.CROP_HEIGHT
        self.x_mm = None
        self.y_mm = None
        self.image_paths = {
            "glass slide": "images/slide carrier_828x662.png",
            "4 glass slide": "images/4 slide carrier_1509x1010.png",
            "6 well plate": "images/6 well plate_1509x1010.png",
            "12 well plate": "images/12 well plate_1509x1010.png",
            "24 well plate": "images/24 well plate_1509x1010.png",
            "96 well plate": "images/96 well plate_1509x1010.png",
            "384 well plate": "images/384 well plate_1509x1010.png",
            "1536 well plate": "images/1536 well plate_1509x1010.png",
        }

        print("navigation viewer:", sample)
        self.init_ui(invertX)

        self.load_background_image(self.image_paths.get(sample, "images/4 slide carrier_1509x1010.png"))
        self.create_layers()
        self.update_display_properties(sample)
        # self.update_display()

    def init_ui(self, invertX):
        # interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground("w")

        self.view = self.graphics_widget.addViewBox(invertX=not INVERTED_OBJECTIVE, invertY=True)
        self.view.setAspectLocked(True)

        self.grid = QVBoxLayout()
        self.grid.addWidget(self.graphics_widget)
        self.setLayout(self.grid)
        # Connect double-click handler
        self.view.scene().sigMouseClicked.connect(self.handle_mouse_click)

    def load_background_image(self, image_path):
        self.view.clear()
        self.background_image = cv2.imread(image_path)
        if self.background_image is None:
            # raise ValueError(f"Failed to load image from {image_path}")
            self.background_image = cv2.imread(self.image_paths.get("glass slide"))

        if len(self.background_image.shape) == 2:  # Grayscale image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_GRAY2RGBA)
        elif self.background_image.shape[2] == 3:  # BGR image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGBA)
        elif self.background_image.shape[2] == 4:  # BGRA image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGRA2RGBA)

        self.background_image_copy = self.background_image.copy()
        self.image_height, self.image_width = self.background_image.shape[:2]
        self.background_item = pg.ImageItem(self.background_image)
        self.view.addItem(self.background_item)

    def create_layers(self):
        self.scan_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        self.fov_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        self.focus_point_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)

        self.scan_overlay_item = pg.ImageItem()
        self.fov_overlay_item = pg.ImageItem()
        self.focus_point_overlay_item = pg.ImageItem()

        self.view.addItem(self.scan_overlay_item)
        self.view.addItem(self.fov_overlay_item)
        self.view.addItem(self.focus_point_overlay_item)

        self.background_item.setZValue(-1)  # Background layer at the bottom
        self.scan_overlay_item.setZValue(0)  # Scan overlay in the middle
        self.fov_overlay_item.setZValue(1)  # FOV overlay next
        self.focus_point_overlay_item.setZValue(2)  # # Focus points on top

    def update_display_properties(self, sample):
        if sample == "glass slide":
            self.location_update_threshold_mm = 0.2
            self.mm_per_pixel = 0.1453
            self.origin_x_pixel = 200
            self.origin_y_pixel = 120
        elif sample == "4 glass slide":
            self.location_update_threshold_mm = 0.2
            self.mm_per_pixel = 0.084665
            self.origin_x_pixel = 50
            self.origin_y_pixel = 0
        else:
            self.location_update_threshold_mm = 0.05
            self.mm_per_pixel = 0.084665
            self.origin_x_pixel = self.a1_x_pixel - (self.a1_x_mm) / self.mm_per_pixel
            self.origin_y_pixel = self.a1_y_pixel - (self.a1_y_mm) / self.mm_per_pixel
        self.update_fov_size()

    def update_fov_size(self):
        pixel_size_um = self.objectiveStore.get_pixel_size()
        self.fov_size_mm = self.acquisition_size * pixel_size_um / 1000

    def on_objective_changed(self):
        self.clear_overlay()
        self.update_fov_size()
        self.draw_current_fov(self.x_mm, self.y_mm)

    def update_wellplate_settings(
        self,
        sample_format,
        a1_x_mm,
        a1_y_mm,
        a1_x_pixel,
        a1_y_pixel,
        well_size_mm,
        well_spacing_mm,
        number_of_skip,
        rows,
        cols,
    ):
        if isinstance(sample_format, QVariant):
            sample_format = sample_format.value()

        if sample_format == "glass slide":
            if IS_HCS:
                sample = "4 glass slide"
            else:
                sample = "glass slide"
        else:
            sample = sample_format

        self.sample = sample
        self.a1_x_mm = a1_x_mm
        self.a1_y_mm = a1_y_mm
        self.a1_x_pixel = a1_x_pixel
        self.a1_y_pixel = a1_y_pixel
        self.well_size_mm = well_size_mm
        self.well_spacing_mm = well_spacing_mm
        self.number_of_skip = number_of_skip
        self.rows = rows
        self.cols = cols

        # Try to find the image for the wellplate
        image_path = self.image_paths.get(sample)
        if image_path is None or not os.path.exists(image_path):
            # Look for a custom wellplate image
            custom_image_path = os.path.join("images", self.sample + ".png")
            print(custom_image_path)
            if os.path.exists(custom_image_path):
                image_path = custom_image_path
            else:
                print(f"Warning: Image not found for {sample}. Using default image.")
                image_path = self.image_paths.get("glass slide")  # Use a default image

        self.load_background_image(image_path)
        self.create_layers()
        self.update_display_properties(sample)
        self.draw_current_fov(self.x_mm, self.y_mm)

    def draw_fov_current_location(self, pos: squid.abc.Pos):
        if not pos:
            if self.x_mm is None and self.y_mm is None:
                return
            self.draw_current_fov(self.x_mm, self.y_mm)
        else:
            x_mm = pos.x_mm
            y_mm = pos.y_mm
            self.draw_current_fov(x_mm, y_mm)
            self.x_mm = x_mm
            self.y_mm = y_mm

    def get_FOV_pixel_coordinates(self, x_mm, y_mm):
        if self.sample == "glass slide":
            current_FOV_top_left = (
                round(self.origin_x_pixel + x_mm / self.mm_per_pixel - self.fov_size_mm / 2 / self.mm_per_pixel),
                round(
                    self.image_height
                    - (self.origin_y_pixel + y_mm / self.mm_per_pixel)
                    - self.fov_size_mm / 2 / self.mm_per_pixel
                ),
            )
            current_FOV_bottom_right = (
                round(self.origin_x_pixel + x_mm / self.mm_per_pixel + self.fov_size_mm / 2 / self.mm_per_pixel),
                round(
                    self.image_height
                    - (self.origin_y_pixel + y_mm / self.mm_per_pixel)
                    + self.fov_size_mm / 2 / self.mm_per_pixel
                ),
            )
        else:
            current_FOV_top_left = (
                round(self.origin_x_pixel + x_mm / self.mm_per_pixel - self.fov_size_mm / 2 / self.mm_per_pixel),
                round((self.origin_y_pixel + y_mm / self.mm_per_pixel) - self.fov_size_mm / 2 / self.mm_per_pixel),
            )
            current_FOV_bottom_right = (
                round(self.origin_x_pixel + x_mm / self.mm_per_pixel + self.fov_size_mm / 2 / self.mm_per_pixel),
                round((self.origin_y_pixel + y_mm / self.mm_per_pixel) + self.fov_size_mm / 2 / self.mm_per_pixel),
            )
        return current_FOV_top_left, current_FOV_bottom_right

    def draw_current_fov(self, x_mm, y_mm):
        self.fov_overlay.fill(0)
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(
            self.fov_overlay, current_FOV_top_left, current_FOV_bottom_right, (255, 0, 0, 255), self.box_line_thickness
        )
        self.fov_overlay_item.setImage(self.fov_overlay)

    def register_fov(self, x_mm, y_mm):
        color = (0, 0, 255, 255)  # Blue RGBA
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(
            self.background_image, current_FOV_top_left, current_FOV_bottom_right, color, self.box_line_thickness
        )
        self.background_item.setImage(self.background_image)

    def register_fov_to_image(self, x_mm, y_mm):
        color = (252, 174, 30, 128)  # Yellow RGBA
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(self.scan_overlay, current_FOV_top_left, current_FOV_bottom_right, color, self.box_line_thickness)
        self.scan_overlay_item.setImage(self.scan_overlay)

    def deregister_fov_to_image(self, x_mm, y_mm):
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(
            self.scan_overlay, current_FOV_top_left, current_FOV_bottom_right, (0, 0, 0, 0), self.box_line_thickness
        )
        self.scan_overlay_item.setImage(self.scan_overlay)

    def register_focus_point(self, x_mm, y_mm):
        """Draw focus point marker as filled circle centered on the FOV"""
        color = (0, 255, 0, 255)  # Green RGBA
        # Get FOV corner coordinates, then calculate FOV center pixel coordinates
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        center_x = (current_FOV_top_left[0] + current_FOV_bottom_right[0]) // 2
        center_y = (current_FOV_top_left[1] + current_FOV_bottom_right[1]) // 2
        # Draw a filled circle at the center
        radius = 5  # Radius of circle in pixels
        cv2.circle(self.focus_point_overlay, (center_x, center_y), radius, color, -1)  # -1 thickness means filled
        self.focus_point_overlay_item.setImage(self.focus_point_overlay)

    def clear_focus_points(self):
        """Clear just the focus point overlay"""
        self.focus_point_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        self.focus_point_overlay_item.setImage(self.focus_point_overlay)

    def clear_slide(self):
        self.background_image = self.background_image_copy.copy()
        self.background_item.setImage(self.background_image)
        self.draw_current_fov(self.x_mm, self.y_mm)

    def clear_overlay(self):
        self.scan_overlay.fill(0)
        self.scan_overlay_item.setImage(self.scan_overlay)
        self.focus_point_overlay.fill(0)
        self.focus_point_overlay_item.setImage(self.focus_point_overlay)

    def handle_mouse_click(self, evt):
        if not evt.double():
            return
        try:
            # Get mouse position in image coordinates (independent of zoom)
            mouse_point = self.background_item.mapFromScene(evt.scenePos())

            # Subtract origin offset before converting to mm
            x_mm = (mouse_point.x() - self.origin_x_pixel) * self.mm_per_pixel
            y_mm = (mouse_point.y() - self.origin_y_pixel) * self.mm_per_pixel

            self._log.debug(f"Got double click at (x_mm, y_mm) = {x_mm, y_mm}")
            self.signal_coordinates_clicked.emit(x_mm, y_mm)

        except Exception as e:
            print(f"Error processing navigation click: {e}")
            return


class ImageArrayDisplayWindow(QMainWindow):

    def __init__(self, window_title=""):
        super().__init__()
        self.setWindowTitle(window_title)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.widget = QWidget()

        # interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.graphics_widget_1 = pg.GraphicsLayoutWidget()
        self.graphics_widget_1.view = self.graphics_widget_1.addViewBox()
        self.graphics_widget_1.view.setAspectLocked(True)
        self.graphics_widget_1.img = pg.ImageItem(border="w")
        self.graphics_widget_1.view.addItem(self.graphics_widget_1.img)
        self.graphics_widget_1.view.invertY()

        self.graphics_widget_2 = pg.GraphicsLayoutWidget()
        self.graphics_widget_2.view = self.graphics_widget_2.addViewBox()
        self.graphics_widget_2.view.setAspectLocked(True)
        self.graphics_widget_2.img = pg.ImageItem(border="w")
        self.graphics_widget_2.view.addItem(self.graphics_widget_2.img)
        self.graphics_widget_2.view.invertY()

        self.graphics_widget_3 = pg.GraphicsLayoutWidget()
        self.graphics_widget_3.view = self.graphics_widget_3.addViewBox()
        self.graphics_widget_3.view.setAspectLocked(True)
        self.graphics_widget_3.img = pg.ImageItem(border="w")
        self.graphics_widget_3.view.addItem(self.graphics_widget_3.img)
        self.graphics_widget_3.view.invertY()

        self.graphics_widget_4 = pg.GraphicsLayoutWidget()
        self.graphics_widget_4.view = self.graphics_widget_4.addViewBox()
        self.graphics_widget_4.view.setAspectLocked(True)
        self.graphics_widget_4.img = pg.ImageItem(border="w")
        self.graphics_widget_4.view.addItem(self.graphics_widget_4.img)
        self.graphics_widget_4.view.invertY()
        ## Layout
        layout = QGridLayout()
        layout.addWidget(self.graphics_widget_1, 0, 0)
        layout.addWidget(self.graphics_widget_2, 0, 1)
        layout.addWidget(self.graphics_widget_3, 1, 0)
        layout.addWidget(self.graphics_widget_4, 1, 1)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)

        # set window size
        desktopWidget = QDesktopWidget()
        width = min(desktopWidget.height() * 0.9, 1000)  # @@@TO MOVE@@@#
        height = width
        self.setFixedSize(int(width), int(height))

    def display_image(self, image, illumination_source):
        if illumination_source < 11:
            self.graphics_widget_1.img.setImage(image, autoLevels=False)
        elif illumination_source == 11:
            self.graphics_widget_2.img.setImage(image, autoLevels=False)
        elif illumination_source == 12:
            self.graphics_widget_3.img.setImage(image, autoLevels=False)
        elif illumination_source == 13:
            self.graphics_widget_4.img.setImage(image, autoLevels=False)


class ConfigurationManager(QObject):
    def __init__(self, filename="channel_configurations.xml"):
        QObject.__init__(self)
        self.config_filename = filename
        self.configurations = []
        self.read_configurations()

    def save_configurations(self):
        self.write_configuration(self.config_filename)

    def write_configuration(self, filename):
        self.config_xml_tree.write(filename, encoding="utf-8", xml_declaration=True, pretty_print=True)

    def read_configurations(self):
        if os.path.isfile(self.config_filename) == False:
            utils_config.generate_default_configuration(self.config_filename)
            print("genenrate default config files")
        self.config_xml_tree = etree.parse(self.config_filename)
        self.config_xml_tree_root = self.config_xml_tree.getroot()
        self.num_configurations = 0
        for mode in self.config_xml_tree_root.iter("mode"):
            self.num_configurations += 1
            self.configurations.append(
                Configuration(
                    mode_id=mode.get("ID"),
                    name=mode.get("Name"),
                    color=self.get_channel_color(mode.get("Name")),
                    exposure_time=float(mode.get("ExposureTime")),
                    analog_gain=float(mode.get("AnalogGain")),
                    illumination_source=int(mode.get("IlluminationSource")),
                    illumination_intensity=float(mode.get("IlluminationIntensity")),
                    camera_sn=mode.get("CameraSN"),
                    z_offset=float(mode.get("ZOffset")),
                    pixel_format=mode.get("PixelFormat"),
                    _pixel_format_options=mode.get("_PixelFormat_options"),
                    emission_filter_position=int(mode.get("EmissionFilterPosition", 1)),
                )
            )

    def update_configuration(self, configuration_id, attribute_name, new_value):
        # Update the XML tree - find ALL modes with this ID
        conf_list = self.config_xml_tree_root.xpath("//mode[contains(@ID," + "'" + str(configuration_id) + "')]")
        for mode_to_update in conf_list:
            mode_to_update.set(attribute_name, str(new_value))
        
        # Also update ALL in-memory configuration objects with this ID
        configs_updated = 0
        for config in self.configurations:
            if config.id == configuration_id:
                if attribute_name == "ExposureTime":
                    old_value = config.exposure_time
                    config.exposure_time = float(new_value)
                    print(f"DEBUG update_configuration: '{config.name}' exposure_time: {old_value} -> {config.exposure_time}")
                    configs_updated += 1
                elif attribute_name == "AnalogGain":
                    config.analog_gain = float(new_value)
                    configs_updated += 1
                elif attribute_name == "IlluminationIntensity":
                    config.illumination_intensity = float(new_value)
                    configs_updated += 1
                elif attribute_name == "IlluminationSource":
                    config.illumination_source = int(new_value)
                    configs_updated += 1
                # Add other attributes as needed
        
        if configs_updated == 0:
            print(f"DEBUG: ERROR - No configurations found with ID '{configuration_id}'!")
            print(f"DEBUG: Available config IDs: {[config.id for config in self.configurations]}")
        elif configs_updated > 1:
            print(f"DEBUG: WARNING - Updated {configs_updated} configurations with duplicate ID '{configuration_id}'")
        
        self.save_configurations()

    def reset_all_exposure_times_to_zero(self):
        """Reset all channel exposure times to 0 (deselected state)"""
        print("DEBUG: Resetting all channel exposure times to 0")
        for config in self.configurations:
            old_exposure = config.exposure_time
            print(f"DEBUG: Calling update_configuration for '{config.name}' with ID '{config.id}'")
            # Use update_configuration which now updates both XML and in-memory object
            self.update_configuration(config.id, "ExposureTime", 0)
            # Check if the update was successful
            if config.exposure_time != 0:
                print(f"DEBUG: ERROR - '{config.name}' failed to reset! Still at {config.exposure_time}ms")
            else:
                print(f"DEBUG: Reset '{config.name}' from {old_exposure}ms to 0ms")
        print(f"DEBUG: Reset {len(self.configurations)} channel exposure times to 0")

    def update_configuration_without_writing(self, configuration_id, attribute_name, new_value):
        conf_list = self.config_xml_tree_root.xpath("//mode[contains(@ID," + "'" + str(configuration_id) + "')]")
        mode_to_update = conf_list[0]
        mode_to_update.set(attribute_name, str(new_value))

    def write_configuration_selected(
        self, selected_configurations, filename
    ):  # to be only used with a throwaway instance
        for conf in self.configurations:
            self.update_configuration_without_writing(conf.id, "Selected", 0)
        for conf in selected_configurations:
            self.update_configuration_without_writing(conf.id, "Selected", 1)
        self.write_configuration(filename)
        for conf in selected_configurations:
            self.update_configuration_without_writing(conf.id, "Selected", 0)

    def get_channel_color(self, channel):
        channel_info = CHANNEL_COLORS_MAP.get(self.extract_wavelength(channel), {"hex": 0xFFFFFF, "name": "gray"})
        return channel_info["hex"]

    def extract_wavelength(self, name):
        # Split the string and find the wavelength number immediately after "Fluorescence"
        parts = name.split()
        if "Fluorescence" in parts:
            index = parts.index("Fluorescence") + 1
            if index < len(parts):
                return parts[index].split()[0]  # Assuming 'Fluorescence 488 nm Ex' and taking '488'
        for color in ["R", "G", "B"]:
            if color in parts or "full_" + color in parts:
                return color
        return None


class ContrastManager:
    def __init__(self):
        self.contrast_limits = {}
        self.acquisition_dtype = None

    def update_limits(self, channel, min_val, max_val):
        self.contrast_limits[channel] = (min_val, max_val)

    def get_limits(self, channel, dtype=None):
        if dtype is not None:
            if self.acquisition_dtype is None:
                self.acquisition_dtype = dtype
            elif self.acquisition_dtype != dtype:
                self.scale_contrast_limits(dtype)
        return self.contrast_limits.get(channel, self.get_default_limits())

    def get_default_limits(self):
        if self.acquisition_dtype is None:
            return (0, 1)
        elif np.issubdtype(self.acquisition_dtype, np.integer):
            info = np.iinfo(self.acquisition_dtype)
            return (info.min, info.max)
        elif np.issubdtype(self.acquisition_dtype, np.floating):
            return (0.0, 1.0)
        else:
            return (0, 1)

    def get_scaled_limits(self, channel, target_dtype):
        min_val, max_val = self.get_limits(channel)
        if self.acquisition_dtype == target_dtype:
            return min_val, max_val

        source_info = np.iinfo(self.acquisition_dtype)
        target_info = np.iinfo(target_dtype)

        scaled_min = (min_val - source_info.min) / (source_info.max - source_info.min) * (
            target_info.max - target_info.min
        ) + target_info.min
        scaled_max = (max_val - source_info.min) / (source_info.max - source_info.min) * (
            target_info.max - target_info.min
        ) + target_info.min

        return scaled_min, scaled_max

    def scale_contrast_limits(self, target_dtype):
        print(f"{self.acquisition_dtype} -> {target_dtype}")
        for channel in self.contrast_limits.keys():
            self.contrast_limits[channel] = self.get_scaled_limits(channel, target_dtype)

        self.acquisition_dtype = target_dtype


class ScanCoordinates(QObject):

    signal_scan_coordinates_updated = Signal()

    def __init__(self, objectiveStore, navigationViewer, stage: AbstractStage):
        QObject.__init__(self)
        self._log = squid.logging.get_logger(self.__class__.__name__)
        # Wellplate settings
        self.objectiveStore = objectiveStore
        self.navigationViewer = navigationViewer
        self.stage = stage
        self.well_selector = None
        self.acquisition_pattern = ACQUISITION_PATTERN
        self.fov_pattern = FOV_PATTERN
        self.format = WELLPLATE_FORMAT
        self.a1_x_mm = A1_X_MM
        self.a1_y_mm = A1_Y_MM
        self.wellplate_offset_x_mm = WELLPLATE_OFFSET_X_mm
        self.wellplate_offset_y_mm = WELLPLATE_OFFSET_Y_mm
        self.well_spacing_mm = WELL_SPACING_MM
        self.well_size_mm = WELL_SIZE_MM
        self.a1_x_pixel = None
        self.a1_y_pixel = None
        self.number_of_skip = None

        # Centralized region management
        self.region_centers = {}  # {region_id: [x, y, z]}
        self.region_shapes = {}  # {region_id: "Square"}
        self.region_fov_coordinates = {}  # {region_id: [(x,y,z), ...]}

    def add_well_selector(self, well_selector):
        self.well_selector = well_selector

    def update_wellplate_settings(
        self, format_, a1_x_mm, a1_y_mm, a1_x_pixel, a1_y_pixel, size_mm, spacing_mm, number_of_skip
    ):
        self.format = format_
        self.a1_x_mm = a1_x_mm
        self.a1_y_mm = a1_y_mm
        self.a1_x_pixel = a1_x_pixel
        self.a1_y_pixel = a1_y_pixel
        self.well_size_mm = size_mm
        self.well_spacing_mm = spacing_mm
        self.number_of_skip = number_of_skip

    def _index_to_row(self, index):
        index += 1
        row = ""
        while index > 0:
            index -= 1
            row = chr(index % 26 + ord("A")) + row
            index //= 26
        return row

    def get_selected_wells(self):
        # get selected wells from the widget
        print("getting selected wells for acquisition")
        if not self.well_selector or self.format == "glass slide":
            return None

        selected_wells = np.array(self.well_selector.get_selected_cells())
        well_centers = {}

        # if no well selected
        if len(selected_wells) == 0:
            return well_centers
        # populate the coordinates
        rows = np.unique(selected_wells[:, 0])
        _increasing = True
        for row in rows:
            items = selected_wells[selected_wells[:, 0] == row]
            columns = items[:, 1]
            columns = np.sort(columns)
            if _increasing == False:
                columns = np.flip(columns)
            for column in columns:
                x_mm = self.a1_x_mm + (column * self.well_spacing_mm) + self.wellplate_offset_x_mm
                y_mm = self.a1_y_mm + (row * self.well_spacing_mm) + self.wellplate_offset_y_mm
                well_id = self._index_to_row(row) + str(column + 1)
                well_centers[well_id] = (x_mm, y_mm)
            _increasing = not _increasing
        return well_centers

    def set_live_scan_coordinates(self, x_mm, y_mm, scan_size_mm, overlap_percent, shape):
        if shape != "Manual" and self.format == "glass slide":
            if self.region_centers:
                self.clear_regions()
            self.add_region("current", x_mm, y_mm, scan_size_mm, overlap_percent, shape)

    def set_well_coordinates(self, scan_size_mm, overlap_percent, shape):
        new_region_centers = self.get_selected_wells()

        if self.format == "glass slide":
            pos = self.stage.get_pos()
            self.set_live_scan_coordinates(pos.x_mm, pos.y_mm, scan_size_mm, overlap_percent, shape)

        elif bool(new_region_centers):
            # Remove regions that are no longer selected
            for well_id in list(self.region_centers.keys()):
                if well_id not in new_region_centers.keys():
                    self.remove_region(well_id)

            # Add regions for selected wells
            for well_id, (x, y) in new_region_centers.items():
                if well_id not in self.region_centers:
                    self.add_region(well_id, x, y, scan_size_mm, overlap_percent, shape)
        else:
            self.clear_regions()

    def set_manual_coordinates(self, manual_shapes, overlap_percent):
        self.clear_regions()
        if manual_shapes is not None:
            # Handle manual ROIs
            manual_region_added = False
            for i, shape_coords in enumerate(manual_shapes):
                scan_coordinates = self.add_manual_region(shape_coords, overlap_percent)
                if scan_coordinates:
                    if len(manual_shapes) <= 1:
                        region_name = f"manual"
                    else:
                        region_name = f"manual{i}"
                    center = np.mean(shape_coords, axis=0)
                    self.region_centers[region_name] = [center[0], center[1]]
                    self.region_shapes[region_name] = "Manual"
                    self.region_fov_coordinates[region_name] = scan_coordinates
                    manual_region_added = True
                    print(f"Added Manual Region: {region_name}")
            if manual_region_added:
                self.signal_scan_coordinates_updated.emit()
        else:
            print("No Manual ROI found")

    def add_region(self, well_id, center_x, center_y, scan_size_mm, overlap_percent=10, shape="Square"):
        """add region based on user inputs"""
        pixel_size_um = self.objectiveStore.get_pixel_size()
        fov_size_mm = (pixel_size_um / 1000) * Acquisition.CROP_WIDTH
        step_size_mm = fov_size_mm * (1 - overlap_percent / 100)
        scan_coordinates = []

        if shape == "Rectangle":
            # Use scan_size_mm as height, width is 0.6 * height
            height_mm = scan_size_mm
            width_mm = scan_size_mm * 0.6
            
            # Calculate steps for height and width separately
            steps_height = math.floor(height_mm / step_size_mm)
            steps_width = math.floor(width_mm / step_size_mm)
            
            # Calculate actual dimensions
            actual_scan_height_mm = (steps_height - 1) * step_size_mm + fov_size_mm
            actual_scan_width_mm = (steps_width - 1) * step_size_mm + fov_size_mm
            
            steps_height = max(1, steps_height)
            steps_width = max(1, steps_width)

            half_steps_height = (steps_height - 1) / 2
            half_steps_width = (steps_width - 1) / 2
            
            for i in range(steps_height):
                row = []
                y = center_y + (i - half_steps_height) * step_size_mm
                for j in range(steps_width):
                    x = center_x + (j - half_steps_width) * step_size_mm
                    if self.validate_coordinates(x, y):
                        row.append((x, y))
                        self.navigationViewer.register_fov_to_image(x, y)
                if self.fov_pattern == "S-Pattern" and i % 2 == 1:
                    row.reverse()
                scan_coordinates.extend(row)
        else:
            steps = math.floor(scan_size_mm / step_size_mm)
            if shape == "Circle":
                tile_diagonal = math.sqrt(2) * fov_size_mm
                if steps % 2 == 1:  # for odd steps
                    actual_scan_size_mm = (steps - 1) * step_size_mm + tile_diagonal
                else:  # for even steps
                    actual_scan_size_mm = math.sqrt(
                        ((steps - 1) * step_size_mm + fov_size_mm) ** 2 + (step_size_mm + fov_size_mm) ** 2
                    )

                if actual_scan_size_mm > scan_size_mm:
                    actual_scan_size_mm -= step_size_mm
                    steps -= 1
            else:
                actual_scan_size_mm = (steps - 1) * step_size_mm + fov_size_mm

            steps = max(1, steps)  # Ensure at least one step
            # print("steps:", steps)
            # print("scan size mm:", scan_size_mm)
            # print("actual scan size mm:", actual_scan_size_mm)
            half_steps = (steps - 1) / 2
            radius_squared = (scan_size_mm / 2) ** 2
            fov_size_mm_half = fov_size_mm / 2

            for i in range(steps):
                row = []
                y = center_y + (i - half_steps) * step_size_mm
                for j in range(steps):
                    x = center_x + (j - half_steps) * step_size_mm
                    if shape == "Square" or shape == "Rectangle" or (
                        shape == "Circle" and self._is_in_circle(x, y, center_x, center_y, radius_squared, fov_size_mm_half)
                    ):
                        if self.validate_coordinates(x, y):
                            row.append((x, y))
                            self.navigationViewer.register_fov_to_image(x, y)

                if self.fov_pattern == "S-Pattern" and i % 2 == 1:
                    row.reverse()
                scan_coordinates.extend(row)

        if not scan_coordinates and shape == "Circle":
            if self.validate_coordinates(center_x, center_y):
                scan_coordinates.append((center_x, center_y))
                self.navigationViewer.register_fov_to_image(center_x, center_y)

        self.region_shapes[well_id] = shape
        self.region_centers[well_id] = [float(center_x), float(center_y), float(self.stage.get_pos().z_mm)]
        self.region_fov_coordinates[well_id] = scan_coordinates
        self.signal_scan_coordinates_updated.emit()
        print(f"Added Region: {well_id}")

    def remove_region(self, well_id):
        if well_id in self.region_centers:
            del self.region_centers[well_id]

            if well_id in self.region_shapes:
                del self.region_shapes[well_id]

            if well_id in self.region_fov_coordinates:
                region_scan_coordinates = self.region_fov_coordinates.pop(well_id)
                for coord in region_scan_coordinates:
                    self.navigationViewer.deregister_fov_to_image(coord[0], coord[1])

            print(f"Removed Region: {well_id}")
            self.signal_scan_coordinates_updated.emit()

    def clear_regions(self):
        self.region_centers.clear()
        self.region_shapes.clear()
        self.region_fov_coordinates.clear()
        self.navigationViewer.clear_overlay()
        self.signal_scan_coordinates_updated.emit()
        print("Cleared All Regions")

    def add_flexible_region(self, region_id, center_x, center_y, center_z, Nx, Ny, overlap_percent=10):
        """Convert grid parameters NX, NY to FOV coordinates based on overlap"""
        fov_size_mm = (self.objectiveStore.get_pixel_size() / 1000) * Acquisition.CROP_WIDTH
        step_size_mm = fov_size_mm * (1 - overlap_percent / 100)

        # Calculate total grid size
        grid_width_mm = (Nx - 1) * step_size_mm
        grid_height_mm = (Ny - 1) * step_size_mm

        scan_coordinates = []
        for i in range(Ny):
            row = []
            y = center_y - grid_height_mm / 2 + i * step_size_mm
            for j in range(Nx):
                x = center_x - grid_width_mm / 2 + j * step_size_mm
                if self.validate_coordinates(x, y):
                    row.append((x, y))
                    self.navigationViewer.register_fov_to_image(x, y)

            if self.fov_pattern == "S-Pattern" and i % 2 == 1:  # reverse even rows
                row.reverse()
            scan_coordinates.extend(row)

        # Region coordinates are already centered since center_x, center_y is grid center
        if scan_coordinates:  # Only add region if there are valid coordinates
            print(f"Added Flexible Region: {region_id}")
            self.region_centers[region_id] = [center_x, center_y, center_z]
            self.region_fov_coordinates[region_id] = scan_coordinates
            self.signal_scan_coordinates_updated.emit()
        else:
            print(f"Region Out of Bounds: {region_id}")

    def add_flexible_region_with_step_size(self, region_id, center_x, center_y, center_z, Nx, Ny, dx, dy):
        """Convert grid parameters NX, NY to FOV coordinates based on dx, dy"""
        grid_width_mm = (Nx - 1) * dx
        grid_height_mm = (Ny - 1) * dy

        # Pre-calculate step sizes and ranges
        x_steps = [center_x - grid_width_mm / 2 + j * dx for j in range(Nx)]
        y_steps = [center_y - grid_height_mm / 2 + i * dy for i in range(Ny)]

        scan_coordinates = []
        for i, y in enumerate(y_steps):
            row = []
            x_range = x_steps if i % 2 == 0 else reversed(x_steps)
            for x in x_range:
                if self.validate_coordinates(x, y):
                    row.append((x, y))
                    self.navigationViewer.register_fov_to_image(x, y)
            scan_coordinates.extend(row)

        if scan_coordinates:  # Only add region if there are valid coordinates
            print(f"Added Flexible Region: {region_id}")
            self.region_centers[region_id] = [center_x, center_y, center_z]
            self.region_fov_coordinates[region_id] = scan_coordinates
            self.signal_scan_coordinates_updated.emit()
        else:
            print(f"Region Out of Bounds: {region_id}")

    def add_manual_region(self, shape_coords, overlap_percent):
        """Add region from manually drawn polygon shape"""
        if shape_coords is None or len(shape_coords) < 3:
            print("Invalid manual ROI data")
            return []

        pixel_size_um = self.objectiveStore.get_pixel_size()
        fov_size_mm = (pixel_size_um / 1000) * Acquisition.CROP_WIDTH
        step_size_mm = fov_size_mm * (1 - overlap_percent / 100)

        # Ensure shape_coords is a numpy array
        shape_coords = np.array(shape_coords)
        if shape_coords.ndim == 1:
            shape_coords = shape_coords.reshape(-1, 2)
        elif shape_coords.ndim > 2:
            print(f"Unexpected shape of manual_shape: {shape_coords.shape}")
            return []

        # Calculate bounding box
        x_min, y_min = np.min(shape_coords, axis=0)
        x_max, y_max = np.max(shape_coords, axis=0)

        # Create a grid of points within the bounding box
        x_range = np.arange(x_min, x_max + step_size_mm, step_size_mm)
        y_range = np.arange(y_min, y_max + step_size_mm, step_size_mm)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))

        # # Use Delaunay triangulation for efficient point-in-polygon test
        # # hull = Delaunay(shape_coords)
        # # mask = hull.find_simplex(grid_points) >= 0
        # # or
        # # Use Ray Casting for point-in-polygon test
        # mask = np.array([self._is_in_polygon(x, y, shape_coords) for x, y in grid_points])

        # # Filter points inside the polygon
        # valid_points = grid_points[mask]

        def corners(x_mm, y_mm, fov):
            center_to_corner = fov/2
            return (
                (x_mm + center_to_corner, y_mm + center_to_corner),
                (x_mm - center_to_corner, y_mm + center_to_corner),
                (x_mm - center_to_corner, y_mm - center_to_corner),
                (x_mm + center_to_corner, y_mm - center_to_corner)
            )
        valid_points = []
        for x_center, y_center in grid_points:
            if not self.validate_coordinates(x_center, y_center):
                self._log.debug(f"Manual coords: ignoring {x_center=},{y_center=} because it is outside our movement range.")
                continue
            if not self._is_in_polygon(x_center, y_center, shape_coords) and not any([self._is_in_polygon(x_corner, y_corner, shape_coords) for (x_corner, y_corner) in corners(x_center, y_center, fov_size_mm)]):
                self._log.debug(f"Manual coords: ignoring {x_center=},{y_center=} because no corners or center are in poly. (corners={corners(x_center, y_center, fov_size_mm)}")
                continue

            valid_points.append((x_center, y_center))
        if not valid_points:
            return []
        valid_points = np.array(valid_points)

        # Sort points
        sorted_indices = np.lexsort((valid_points[:, 0], valid_points[:, 1]))
        sorted_points = valid_points[sorted_indices]

        # Apply S-Pattern if needed
        if self.fov_pattern == "S-Pattern":
            unique_y = np.unique(sorted_points[:, 1])
            for i in range(1, len(unique_y), 2):
                mask = sorted_points[:, 1] == unique_y[i]
                sorted_points[mask] = sorted_points[mask][::-1]

        # Register FOVs
        for x, y in sorted_points:
            self.navigationViewer.register_fov_to_image(x, y)

        return sorted_points.tolist()

    def region_contains_coordinate(self, region_id: str, x: float, y: float) -> bool:
        # TODO: check for manual region
        if not self.validate_region(region_id):
            return False

        bounds = self.get_region_bounds(region_id)
        shape = self.get_region_shape(region_id)

        # For square regions
        if not (bounds["min_x"] <= x <= bounds["max_x"] and bounds["min_y"] <= y <= bounds["max_y"]):
            return False

        # For circle regions
        if shape == "Circle":
            center_x = (bounds["max_x"] + bounds["min_x"]) / 2
            center_y = (bounds["max_y"] + bounds["min_y"]) / 2
            radius = (bounds["max_x"] - bounds["min_x"]) / 2
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius**2:
                return False

        return True

    def _is_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _is_in_circle(self, x, y, center_x, center_y, radius_squared, fov_size_mm_half):
        corners = [
            (x - fov_size_mm_half, y - fov_size_mm_half),
            (x + fov_size_mm_half, y - fov_size_mm_half),
            (x - fov_size_mm_half, y + fov_size_mm_half),
            (x + fov_size_mm_half, y + fov_size_mm_half),
        ]
        return all((cx - center_x) ** 2 + (cy - center_y) ** 2 <= radius_squared for cx, cy in corners)

    def has_regions(self):
        """Check if any regions exist"""
        return len(self.region_centers) > 0

    def validate_region(self, region_id):
        """Validate a region exists"""
        return region_id in self.region_centers and region_id in self.region_fov_coordinates

    def validate_coordinates(self, x, y):
        return (
            SOFTWARE_POS_LIMIT.X_NEGATIVE <= x <= SOFTWARE_POS_LIMIT.X_POSITIVE
            and SOFTWARE_POS_LIMIT.Y_NEGATIVE <= y <= SOFTWARE_POS_LIMIT.Y_POSITIVE
        )

    def sort_coordinates(self):
        print(f"Acquisition pattern: {self.acquisition_pattern}")

        if len(self.region_centers) <= 1:
            return

        def sort_key(item):
            key, coord = item
            if "manual" in key:
                return (0, coord[1], coord[0])  # Manual coords: sort by y, then x
            else:
                row, col = key[0], int(key[1:])
                return (1, ord(row), col)  # Well coords: sort by row, then column

        sorted_items = sorted(self.region_centers.items(), key=sort_key)

        if self.acquisition_pattern == "S-Pattern":
            # Group by row and reverse alternate rows
            rows = itertools.groupby(sorted_items, key=lambda x: x[1][1] if "manual" in x[0] else x[0][0])
            sorted_items = []
            for i, (_, group) in enumerate(rows):
                row = list(group)
                if i % 2 == 1:
                    row.reverse()
                sorted_items.extend(row)

        # Update dictionaries efficiently
        self.region_centers = {k: v for k, v in sorted_items}
        self.region_fov_coordinates = {
            k: self.region_fov_coordinates[k] for k, _ in sorted_items if k in self.region_fov_coordinates
        }

    def get_region_bounds(self, region_id):
        """Get region boundaries"""
        if not self.validate_region(region_id):
            return None
        fovs = np.array(self.region_fov_coordinates[region_id])
        return {
            "min_x": np.min(fovs[:, 0]),
            "max_x": np.max(fovs[:, 0]),
            "min_y": np.min(fovs[:, 1]),
            "max_y": np.max(fovs[:, 1]),
        }

    def get_region_shape(self, region_id):
        if not self.validate_region(region_id):
            return None
        return self.region_shapes[region_id]

    def get_scan_bounds(self):
        """Get bounds of all scan regions with margin"""
        if not self.has_regions():
            return None

        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        # Find global bounds across all regions
        for region_id in self.region_fov_coordinates.keys():
            bounds = self.get_region_bounds(region_id)
            if bounds:
                min_x = min(min_x, bounds["min_x"])
                max_x = max(max_x, bounds["max_x"])
                min_y = min(min_y, bounds["min_y"])
                max_y = max(max_y, bounds["max_y"])

        if min_x == float("inf"):
            return None

        # Add margin around bounds (5% of larger dimension)
        width = max_x - min_x
        height = max_y - min_y
        margin = max(width, height) * 0.00  # 0.05

        return {"x": (min_x - margin, max_x + margin), "y": (min_y - margin, max_y + margin)}

    def update_fov_z_level(self, region_id, fov, new_z):
        """Update z-level for a specific FOV and its region center"""
        if not self.validate_region(region_id):
            print(f"Region {region_id} not found")
            return

        # Update FOV coordinates
        fov_coords = self.region_fov_coordinates[region_id]
        if fov < len(fov_coords):
            # Handle both (x,y) and (x,y,z) cases
            x, y = fov_coords[fov][:2]  # Takes first two elements regardless of length
            self.region_fov_coordinates[region_id][fov] = (x, y, new_z)

        # If first FOV, update region center coordinates
        if fov == 0:
            if len(self.region_centers[region_id]) == 3:
                self.region_centers[region_id][2] = new_z
            else:
                self.region_centers[region_id].append(new_z)

        print(f"Updated z-level to {new_z} for region:{region_id}, fov:{fov}")


from scipy.interpolate import SmoothBivariateSpline, RBFInterpolator


class FocusMap:
    """Handles fitting and interpolation of slide surfaces through measured focus points"""

    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self.surface_fit = None
        self.method = "spline"  # can be 'spline' or 'rbf'
        self.is_fitted = False
        self.points_xyz = None

    def generate_grid_coordinates(
        self, scanCoordinates: ScanCoordinates, rows: int = 4, cols: int = 4, add_margin: bool = False
    ) -> List[Tuple[float, float]]:
        """
        Generate focus point grid coordinates for each scan region

        Args:
            scanCoordinates: ScanCoordinates instance containing regions
            rows: Number of rows in focus grid
            cols: Number of columns in focus grid
            add_margin: If True, adds margin to avoid points at region borders

        Returns:
            list of (x,y) coordinate tuples for focus points
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Number of rows and columns must be greater than 0")

        focus_points = []

        # Generate focus points for each region
        for region_id, region_coords in scanCoordinates.region_fov_coordinates.items():
            # Get region bounds
            bounds = scanCoordinates.get_region_bounds(region_id)
            if not bounds:
                continue

            x_min, x_max = bounds["min_x"], bounds["max_x"]
            y_min, y_max = bounds["min_y"], bounds["max_y"]

            # For add_margin we are using one more row and col, taking the middle points on the grid so that the
            # focus points are not located at the edges of the scaning grid.
            # TODO: set a value for margin from user input
            if add_margin:
                x_step = (x_max - x_min) / cols if cols > 1 else 0
                y_step = (y_max - y_min) / rows if rows > 1 else 0
            else:
                x_step = (x_max - x_min) / (cols - 1) if cols > 1 else 0
                y_step = (y_max - y_min) / (rows - 1) if rows > 1 else 0

            # Generate grid points
            for i in range(rows):
                for j in range(cols):
                    if add_margin:
                        x = x_min + x_step / 2 + j * x_step
                        y = y_min + y_step / 2 + i * y_step
                    else:
                        x = x_min + j * x_step
                        y = y_min + i * y_step

                    # Check if point is within region bounds
                    if scanCoordinates.validate_coordinates(x, y) and scanCoordinates.region_contains_coordinate(
                        region_id, x, y
                    ):
                        focus_points.append((x, y))

        return focus_points

    def set_method(self, method):
        """Set interpolation method

        Args:
            method (str): Either 'spline' or 'rbf' (Radial Basis Function)
        """
        if method not in ["spline", "rbf"]:
            raise ValueError("Method must be either 'spline' or 'rbf'")
        self.method = method
        self.is_fitted = False

    def fit(self, points):
        """Fit surface through provided focus points

        Args:
            points (list): List of (x,y,z) tuples

        Returns:
            tuple: (mean_error, std_error) in mm
        """
        if len(points) < 4:
            raise ValueError("Need at least 4 points to fit surface")

        self.points = np.array(points)
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        if self.method == "spline":
            try:
                self.surface_fit = SmoothBivariateSpline(
                    x, y, z, kx=3, ky=3, s=self.smoothing_factor  # cubic spline in x  # cubic spline in y
                )
            except Exception as e:
                print(f"Spline fitting failed: {str(e)}, falling back to RBF")
                self.method = "rbf"
                self._fit_rbf(x, y, z)
        else:
            self._fit_rbf(x, y, z)

        self.is_fitted = True
        errors = self._calculate_fitting_errors()
        return np.mean(errors), np.std(errors)

    def _fit_rbf(self, x, y, z):
        """Fit using Radial Basis Function interpolation"""
        xy = np.column_stack((x, y))
        self.surface_fit = RBFInterpolator(xy, z, kernel="thin_plate_spline", epsilon=self.smoothing_factor)

    def interpolate(self, x, y):
        """Get interpolated Z value at given (x,y) coordinates

        Args:
            x (float or array): X coordinate(s)
            y (float or array): Y coordinate(s)

        Returns:
            float or array: Interpolated Z value(s)
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit surface before interpolating")

        if np.isscalar(x) and np.isscalar(y):
            if self.method == "spline":
                return float(self.surface_fit.ev(x, y))
            else:
                return float(self.surface_fit([[x, y]]))
        else:
            x = np.asarray(x)
            y = np.asarray(y)
            if self.method == "spline":
                return self.surface_fit.ev(x, y)
            else:
                xy = np.column_stack((x.ravel(), y.ravel()))
                z = self.surface_fit(xy)
                return z.reshape(x.shape)

    def _calculate_fitting_errors(self):
        """Calculate absolute errors at measured points"""
        errors = []
        for x, y, z_measured in self.points:
            z_fit = self.interpolate(x, y)
            errors.append(abs(z_fit - z_measured))
        return np.array(errors)

    def get_surface_grid(self, x_range, y_range, num_points=50):
        """Generate grid of interpolated Z values for visualization

        Args:
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            num_points (int): Number of points per dimension

        Returns:
            tuple: (X grid, Y grid, Z grid)
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit surface before generating grid")

        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        Z = self.interpolate(X, Y)

        return X, Y, Z


class LaserAutofocusController(QObject):

    image_to_display = Signal(np.ndarray)
    signal_displacement_um = Signal(float)

    def __init__(
        self,
        microcontroller: Microcontroller,
        camera,
        liveController,
        stage: AbstractStage,
        has_two_interfaces=True,
        use_glass_top=True,
        look_for_cache=True,
    ):
        QObject.__init__(self)
        self.microcontroller = microcontroller
        self.camera = camera
        self.liveController = liveController
        self.stage = stage

        self.is_initialized = False
        self.x_reference = 0
        self.pixel_to_um = 1
        self.x_offset = 0
        self.y_offset = 0
        self.x_width = 3088
        self.y_width = 2064

        self.has_two_interfaces = has_two_interfaces  # e.g. air-glass and glass water, set to false when (1) using oil immersion (2) using 1 mm thick slide (3) using metal coated slide or Si wafer
        self.use_glass_top = use_glass_top
        self.spot_spacing_pixels = None  # spacing between the spots from the two interfaces (unit: pixel)

        self.look_for_cache = look_for_cache

        self.image = None  # for saving the focus camera image for debugging when centroid cannot be found

        if look_for_cache:
            cache_path = "cache/laser_af_reference_plane.txt"
            try:
                with open(cache_path, "r") as cache_file:
                    for line in cache_file:
                        value_list = line.split(",")
                        x_offset = float(value_list[0])
                        y_offset = float(value_list[1])
                        width = int(value_list[2])
                        height = int(value_list[3])
                        pixel_to_um = float(value_list[4])
                        x_reference = float(value_list[5])
                        self.initialize_manual(x_offset, y_offset, width, height, pixel_to_um, x_reference)
                        break
            except (FileNotFoundError, ValueError, IndexError) as e:
                print("Unable to read laser AF state cache, exception below:")
                print(e)
                pass

    def initialize_manual(self, x_offset, y_offset, width, height, pixel_to_um, x_reference, write_to_cache=True):
        cache_string = ",".join(
            [str(x_offset), str(y_offset), str(width), str(height), str(pixel_to_um), str(x_reference)]
        )
        if write_to_cache:
            cache_path = Path("cache/laser_af_reference_plane.txt")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(cache_string)
        # x_reference is relative to the full sensor
        self.pixel_to_um = pixel_to_um
        self.x_offset = int((x_offset // 8) * 8)
        self.y_offset = int((y_offset // 2) * 2)
        self.width = int((width // 8) * 8)
        self.height = int((height // 2) * 2)
        self.x_reference = x_reference - self.x_offset  # self.x_reference is relative to the cropped region
        self.camera.set_ROI(self.x_offset, self.y_offset, self.width, self.height)
        self.is_initialized = True

    def initialize_auto(self):

        # first find the region to crop
        # then calculate the convert factor

        # set camera to use full sensor
        self.camera.set_ROI(0, 0, None, None)  # set offset first
        self.camera.set_ROI(0, 0, 3088, 2064)
        # update camera settings
        self.camera.set_exposure_time(FOCUS_CAMERA_EXPOSURE_TIME_MS)
        self.camera.set_analog_gain(FOCUS_CAMERA_ANALOG_GAIN)

        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        # get laser spot location
        x, y = self._get_laser_spot_centroid()

        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        x_offset = x - LASER_AF_CROP_WIDTH / 2
        y_offset = y - LASER_AF_CROP_HEIGHT / 2
        print("laser spot location on the full sensor is (" + str(int(x)) + "," + str(int(y)) + ")")

        # set camera crop
        self.initialize_manual(x_offset, y_offset, LASER_AF_CROP_WIDTH, LASER_AF_CROP_HEIGHT, 1, x)

        # turn on laser
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        # move z to - 6 um
        self.stage.move_z(-0.018)
        self.stage.move_z(0.012)
        time.sleep(0.02)

        # measure
        x0, y0 = self._get_laser_spot_centroid()

        # move z to 6 um
        self.stage.move_z(0.006)
        time.sleep(0.02)

        # measure
        x1, y1 = self._get_laser_spot_centroid()

        # turn off laser
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        if x1 - x0 == 0:
            # for simulation
            self.pixel_to_um = 0.4
        else:
            # calculate the conversion factor
            self.pixel_to_um = 6.0 / (x1 - x0)
        print("pixel to um conversion factor is " + str(self.pixel_to_um) + " um/pixel")

        # set reference
        self.x_reference = x1

        if self.look_for_cache:
            cache_path = "cache/laser_af_reference_plane.txt"
            try:
                x_offset = None
                y_offset = None
                width = None
                height = None
                pixel_to_um = None
                x_reference = None
                with open(cache_path, "r") as cache_file:
                    for line in cache_file:
                        value_list = line.split(",")
                        x_offset = float(value_list[0])
                        y_offset = float(value_list[1])
                        width = int(value_list[2])
                        height = int(value_list[3])
                        pixel_to_um = self.pixel_to_um
                        x_reference = self.x_reference + self.x_offset
                        break
                cache_string = ",".join(
                    [str(x_offset), str(y_offset), str(width), str(height), str(pixel_to_um), str(x_reference)]
                )
                cache_path = Path("cache/laser_af_reference_plane.txt")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(cache_string)
            except (FileNotFoundError, ValueError, IndexError) as e:
                print("Unable to read laser AF state cache, exception below:")
                print(e)
                pass

    def measure_displacement(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()
        # get laser spot location
        x, y = self._get_laser_spot_centroid()
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()
        # calculate displacement
        displacement_um = (x - self.x_reference) * self.pixel_to_um
        self.signal_displacement_um.emit(displacement_um)
        return displacement_um

    def move_to_target(self, target_um):
        current_displacement_um = self.measure_displacement()
        print("Laser AF displacement: ", current_displacement_um)

        if abs(current_displacement_um) > LASER_AF_RANGE:
            print(
                f"Warning: Measured displacement ({current_displacement_um:.1f} μm) is unreasonably large, using previous z position"
            )
            um_to_move = 0
        else:
            um_to_move = target_um - current_displacement_um

        self.stage.move_z(um_to_move / 1000)

        # update the displacement measurement
        self.measure_displacement()

    def set_reference(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()
        # get laser spot location
        x, y = self._get_laser_spot_centroid()
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()
        self.x_reference = x
        self.signal_displacement_um.emit(0)

    def _caculate_centroid(self, image):
        if self.has_two_interfaces == False:
            h, w = image.shape
            x, y = np.meshgrid(range(w), range(h))
            I = image.astype(float)
            I = I - np.amin(I)
            I[I / np.amax(I) < 0.2] = 0
            x = np.sum(x * I) / np.sum(I)
            y = np.sum(y * I) / np.sum(I)
            return x, y
        else:
            I = image
            # get the y position of the spots
            tmp = np.sum(I, axis=1)
            y0 = np.argmax(tmp)
            # crop along the y axis
            I = I[y0 - 96 : y0 + 96, :]
            # signal along x
            tmp = np.sum(I, axis=0)
            # find peaks
            peak_locations, _ = scipy.signal.find_peaks(tmp, distance=100)
            idx = np.argsort(tmp[peak_locations])
            peak_0_location = peak_locations[idx[-1]]
            peak_1_location = peak_locations[
                idx[-2]
            ]  # for air-glass-water, the smaller peak corresponds to the glass-water interface
            self.spot_spacing_pixels = peak_1_location - peak_0_location
            """
            # find peaks - alternative
            if self.spot_spacing_pixels is not None:
                peak_locations,_ = scipy.signal.find_peaks(tmp,distance=100)
                idx = np.argsort(tmp[peak_locations])
                peak_0_location = peak_locations[idx[-1]]
                peak_1_location = peak_locations[idx[-2]] # for air-glass-water, the smaller peak corresponds to the glass-water interface
                self.spot_spacing_pixels = peak_1_location-peak_0_location
            else:
                peak_0_location = np.argmax(tmp)
                peak_1_location = peak_0_location + self.spot_spacing_pixels
            """
            # choose which surface to use
            if self.use_glass_top:
                x1 = peak_1_location
            else:
                x1 = peak_0_location
            # find centroid
            h, w = I.shape
            x, y = np.meshgrid(range(w), range(h))
            I = I[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            x = x[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            y = y[:, max(0, x1 - 64) : min(w - 1, x1 + 64)]
            I = I.astype(float)
            I = I - np.amin(I)
            I[I / np.amax(I) < 0.1] = 0
            x1 = np.sum(x * I) / np.sum(I)
            y1 = np.sum(y * I) / np.sum(I)
            return x1, y0 - 96 + y1

    def _get_laser_spot_centroid(self):
        # disable camera callback
        self.camera.disable_callback()
        tmp_x = 0
        tmp_y = 0
        for i in range(LASER_AF_AVERAGING_N):
            # send camera trigger
            if self.liveController.trigger_mode == TriggerMode.SOFTWARE:
                self.camera.send_trigger()
            elif self.liveController.trigger_mode == TriggerMode.HARDWARE:
                # self.microcontroller.send_hardware_trigger(control_illumination=True,illumination_on_time_us=self.camera.exposure_time*1000)
                pass  # to edit
            # read camera frame
            image = self.camera.read_frame()
            self.image = image
            # optionally display the image
            if LASER_AF_DISPLAY_SPOT_IMAGE:
                self.image_to_display.emit(image)
            # calculate centroid
            x, y = self._caculate_centroid(image)
            tmp_x = tmp_x + x
            tmp_y = tmp_y + y
        x = tmp_x / LASER_AF_AVERAGING_N
        y = tmp_y / LASER_AF_AVERAGING_N
        return x, y

    def get_image(self):
        # turn on the laser
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()  # send trigger, grab image and display image
        self.camera.send_trigger()
        image = self.camera.read_frame()
        self.image_to_display.emit(image)
        # turn off the laser
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()
        return image
