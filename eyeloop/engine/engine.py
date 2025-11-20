"""Core engine for eyeloop eye-tracking module."""

import logging
import time
from os.path import dirname, abspath
import glob
import os

import numpy as np

import eyeloop.config as config
from eyeloop.constants.engine_constants import *
from eyeloop.engine.processor import Shape
from eyeloop.utilities.paramSave import save_pupil_parameters
from eyeloop.utilities.paramRead import read_pupil_parameters
from vr_core.utilities.logger_setup import setup_logger


PARAMS_DIR = f"{dirname(dirname(abspath(__file__)))}/engine/params"

class Engine:
    """Core engine for eyeloop eye-tracking module."""

    def __init__(self, eyeloop, extractor):

        self.logger = setup_logger("EyeLoop_Engine")
        self.extractor = extractor
        self.live = True  # Access this to check if Core is running.

        self.eyeloop = eyeloop
        self.model = config.arguments.model  # Used for assigning appropriate circular model.
        self.side = config.arguments.side

        self.iterate = self.track

        self.angle = 0
        self.width: int
        self.height: int
        self.center: tuple[int, int]

        self.pupil_processor = Shape()


    def arm(self, width, height, image) -> None:
        """Arms the engine with initial parameters and settings."""

        self.width, self.height = width, height

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.arm(width, height)

        self.center = (width//2, height//2)

        self.iterate(image)

        if config.arguments.blinkcalibration != "":
            config.blink = np.load(config.arguments.blinkcalibration)
            self.blink_sampled = lambda _:None
            #logger.info("(success) blink calibration loaded")


    def blink_sampled(self, t: int = 1):
        """Calibrates blink detection based on sampled mean image intensity."""

        if t == 1:
            pass
            #if config.blink_i % 20 == 0:
                # print(f"calibrating blink detector "
                #     f"{round(config.blink_i/config.blink.shape[0]*100,1)}%")
        else:
            self.logger.info("(success) blink detection calibrated")
            path = f"{config.file_manager.new_folderpath}/blinkcalibration_.npy"
            np.save(path, config.blink)
            print("blink calibration file saved")


    def track(self, img) -> None:
        """
        Executes the tracking algorithm on the pupil and corneal reflections.
        First, blinking is analyzed.
        Second, corneal reflections are detected.
        Third, corneal reflections are inverted at pupillary overlap.
        Fourth, pupil is detected.
        Finally, data is logged and extractors are run.
        """
        mean_img = np.mean(img)

        try:
            config.blink[config.blink_i] = mean_img
            config.blink_i += 1
            self.blink_sampled(1)

        except IndexError:
            self.blink_sampled(0)
            self.blink_sampled = lambda _: None
            config.blink_i = 0

        self.dataout = {}

        if np.abs(mean_img - np.mean(config.blink[np.nonzero(config.blink)])) > 10:
            #self.dataout["blink"] = 1
            self.pupil_processor.fit_model.params = None
            #logger.info("Blink detected.")
        else:
            self.pupil_processor.track(img)

        if config.arguments.use_gui == 1:
            try:
                config.graphical_user_interface.update(img)
            except Exception as e:
                print("Did you assign the graphical user interface (GUI) correctly? "
                    f"Attempting to release(): {e}")
                self.release()
                return

        self.extractor.fetch(self)


    def release(self) -> None:
        """
        Releases/deactivates all running process, i.e., importers, extractors.
        """
        self.live = False

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.release()
            self.extractor.release(self)
            config.importer.release()
