"""Core engine for eyeloop eye-tracking module."""

from typing import Any
from numpy import diff
import eyeloop.config as config
# ruff: noqa F403
from eyeloop.constants.engine_constants import *
from eyeloop.engine.processor import Shape
from vr_core.utilities.logger_setup import setup_logger


class Engine:
    """Core engine for eyeloop eye-tracking module."""

    def __init__(self, eyeloop, extractor):

        self.side = config.arguments.side
        self.logger = setup_logger(f"{self.side} engine")
        self.extractor = extractor
        self.live = True  # Access this to check if Core is running.

        self.process_blink = False

        self.eyeloop = eyeloop
        self.model = config.arguments.model  # Used for assigning appropriate circular model.

        self.angle = 0
        self.width: int
        self.height: int
        self.center: tuple[int, int]

        self.pupil_processor = Shape(track_type="pupil")
        self.cr_processor = Shape(track_type="cr")
        # self.cr_processor = None

        # Initialize dataout attribute
        self.dataout: dict[str, Any] = {}

    def arm(self, width, height, image) -> None:
        """Arms the engine with initial parameters and settings."""

        self.width, self.height = width, height

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.arm(width, height)

        self.center = (width//2, height//2)

        self.track(image)


    def track(self, img) -> None:
        """
        Executes the tracking algorithm on the pupil and corneal reflections.
        First, blinking is analyzed.
        Second, corneal reflections are detected.
        Third, corneal reflections are inverted at pupillary overlap.
        Fourth, pupil is detected.
        Finally, data is logged and extractors are run.
        """
        if self.process_blink:
            mean_img = np.mean(img)

            try:
                config.blink[config.blink_i] = mean_img
                config.blink_i += 1

            except IndexError:

                config.blink_i = 0

            baseline = np.mean(config.blink[np.nonzero(config.blink)])
            diff = np.abs(mean_img - baseline)

        self.dataout["pupil"] = ()
        self.dataout["cr"] = ()

        if self.process_blink and diff > 3:
            self.dataout["pupil"] = ()
            self.dataout["cr"] = ()
            self.logger.info("Blink detected.")
        else:
            self.pupil_processor.track(img)
            if self.cr_processor is not None:
                self.cr_processor.track(img)

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.update(img)

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
