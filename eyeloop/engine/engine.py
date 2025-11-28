# ruff: noqa: ERA001, ANN401

"""Core engine for eyeloop eye-tracking module."""

import time
from typing import Any

import numpy as np
from eyeloop import config

# ruff: noqa: F403
from eyeloop.constants.engine_constants import *
from eyeloop.engine.processor import Shape

from vr_core.utilities.logger_setup import setup_logger


class Engine:
    """Core engine for eyeloop eye-tracking module."""

    def __init__(self, eyeloop: Any, extractor: Any) -> None:
        """Initialize the engine with necessary components."""
        self.side = config.arguments.side
        self.logger = setup_logger(f"{self.side} engine")
        self.extractor = extractor
        self.live = True  # Access this to check if Core is running.

        print_status = config.arguments.eng_profiling

        if print_status in {"Both", config.arguments.side}:
            self.profile_enabled = True
        else:
            self.profile_enabled = False

        self.process_blink = False
        self.brightness_threshold = 3

        self.eyeloop = eyeloop
        self.model = config.arguments.model  # Used for assigning appropriate circular model.

        self.width: int
        self.height: int
        self.center: tuple[int, int]

        self.pupil_processor = Shape(track_type="pupil")
        self.cr_processor = Shape(track_type="cr")
        # self.cr_processor = None

        # Initialize dataout attribute
        self.dataout: dict[str, Any] = {}
        self.timing_cycle: int = 0
        self.print_cycle: int = 100

        self.time_start_mid: float = 0.0
        self.time_mid_end: float = 0.0
        self.time_end_gui_fetch: float = 0.0
        self.time_total: float = 0.0

        self.time_eng_fps: float = 0.0

        self.time_import: float = 0.0
        self.time_import1: float = 0.0


    def arm(self, width: int, height: int, image: Any) -> None:
        """Arms the engine with initial parameters and settings."""
        self.width, self.height = width, height

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.arm(width, height)

        self.center = (width//2, height//2)

        self.track(image)


    def track(self, img: Any) -> None:
        """Execute the tracking algorithm on the pupil and corneal reflections.

        First, blinking is analyzed.
        Second, corneal reflections are detected.
        Third, corneal reflections are inverted at pupillary overlap.
        Fourth, pupil is detected.
        Finally, data is logged and extractors are run.
        """
        time_start = time.perf_counter_ns() / 1e9

        if self.timing_cycle != 0:
            self.time_import += time.perf_counter_ns() / 1e9 - self.time_import1

        self.timing_cycle += 1

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

        if self.process_blink and diff > self.brightness_threshold:
            self.dataout["pupil"] = ()
            self.dataout["cr"] = ()
            self.logger.info("Blink detected.")
        else:
            self.pupil_processor.track(img)

            time_mid = time.perf_counter_ns() / 1e9
            self.time_start_mid += time_mid - time_start

            if self.cr_processor is not None:
                self.cr_processor.track(img)

            time_end = time.perf_counter_ns() / 1e9
            self.time_mid_end += time_end - time_mid

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.update(img)

        self.extractor.fetch(self)

        time_gui_fetch = time.perf_counter_ns() / 1e9
        self.time_end_gui_fetch += time_gui_fetch - time_end

        self.time_total += time_gui_fetch - time_start

        self.time_import1 = time.perf_counter_ns() / 1e9

        if self.timing_cycle == self.print_cycle:
            self._log_timings()
            self.timing_cycle = 0


    def release(self) -> None:
        """Release/deactivate all running process, i.e., importers, extractors."""
        self.live = False

        if config.arguments.use_gui == 1:
            config.graphical_user_interface.release()
            self.extractor.release(self)
            config.importer.release()


    def _log_timings(self) -> None:
        """Log timing information for the tracking process."""
        if self.profile_enabled:
            self.logger.info(
                "PU: %.3fms; CR: %.3fms; EX: %.3fms; T: %.1fms; E-FPS: %f; IM: %.3f; T-FPS: %f",
                (self.time_start_mid / self.print_cycle) * 1000,
                (self.time_mid_end / self.print_cycle) * 1000,
                (self.time_end_gui_fetch / self.print_cycle) * 1000,
                (self.time_total / self.print_cycle) * 1000,
                self.print_cycle / (self.time_total),
                (self.time_import / self.print_cycle) * 1000,
                self.print_cycle / (self.time_total + self.time_import),
            )
        self.time_import = 0.0
        self.time_start_mid = 0.0
        self.time_mid_end = 0.0
        self.time_end_gui_fetch = 0.0
        self.time_total = 0.0
