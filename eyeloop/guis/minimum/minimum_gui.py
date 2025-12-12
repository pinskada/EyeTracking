# ruff: noqa: TRY400, BLE001

"""Minimum GUI for Eyeloop module."""

import cv2
import numpy as np
from eyeloop import config
from eyeloop.constants.minimum_gui_constants import *  # noqa: F403
from eyeloop.utilities.general_operations import to_int, tuple_int

from vr_core.utilities.logger_setup import setup_logger


class GUI:
    """Minimum GUI for Eyeloop module."""

    def __init__(self) -> None:
        """Initialize the GUI with necessary components."""
        self.logger = setup_logger("Eyeloop GUI")

        self.side = config.arguments.side

        self.preview = self.side + "_preview"
        self.pupil_bin = self.side + "_binary_pupil"
        self.cr_bin = self.side + "_binary_cr"

        self.print_cycle = 0
        self.print_fps = 5


    def release(self) -> None:
        """Release GUI resources."""
        cv2.destroyAllWindows()


    def arm(self, width: int, height: int) -> None:
        """Arm the GUI with initial parameters and settings."""
        self.pupil_processor = config.engine.pupil_processor

        self.bin_stock = np.zeros((height, width))
        self.bin_P = self.bin_stock.copy()

        scale = 0.5

        width = int(np.floor(width * scale))
        height = int(np.floor(height * scale))

        if (self.side == "Right"):
            x_shift = width
            x_shift_cr_bin = 3 * width
        else:
            x_shift = 0
            x_shift_cr_bin = 2 * width

        cv2.namedWindow(self.preview, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.pupil_bin, cv2.WINDOW_NORMAL)


        self.logger.info("%s eye; width: %s; height: %s", self.side, width, height)
        cv2.resizeWindow(self.preview, width, height)
        cv2.resizeWindow(self.pupil_bin, width, height)

        cv2.moveWindow(self.preview, x_shift, 20)
        cv2.moveWindow(self.pupil_bin, x_shift, height + 50)

        if config.engine.cr_processor is not None:
            cv2.namedWindow(self.cr_bin, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cr_bin, width, height)
            cv2.moveWindow(self.cr_bin, x_shift_cr_bin, height + 50)


    def place_cross(
        self,
        source: np.ndarray,
        center: tuple[float, float],
        color: tuple[float, float, float],
        thickness: int,
        size: int,
    ) -> None:
        """Place a cross at the specified center on the source image."""
        try:
            source[
                to_int(center[1] - size):to_int(center[1] + size-1),
                to_int(center[0]-thickness):to_int(center[0]+thickness),
            ] = color
            source[
                to_int(center[1]-thickness):to_int(center[1]+thickness),
                to_int(center[0] - size):to_int(center[0] + size-1),
            ] = color
        except Exception:
            self.logger.error("Cross placement error at center: %s", center)


    def draw(self, source_rgb: np.ndarray) -> None:
        """Draw pupil and CR marks on the source image."""
        if config.engine.dataout["pupil"]:
            try:
                pp = config.engine.dataout["pupil"]
                cv2.ellipse(
                    source_rgb,
                    tuple_int(pp.center),
                    tuple_int((pp.radius, pp.radius)),
                    0, 0, 360, red, 1, # noqa: F405
                )
                radius = config.engine.cr_processor.distance_transform.mask_radius
                cv2.ellipse(
                        source_rgb,
                        tuple_int(pp.center),
                        tuple_int((radius, radius)),
                        0, 0, 360, blue, 1, # noqa: F405
                    )
                self.place_cross(source_rgb, pp.center, red, 1, 20)  # noqa: F405
            except Exception as e:
                self.logger.error("Pupil mark error: %s", e)

        if config.engine.dataout["cr"]:
            try:
                cr_list = config.engine.dataout["cr"]
                for cr in cr_list:
                    color = pink if cr.is_filled else green  # noqa: F405
                    self.place_cross(source_rgb, cr.center, color, 2, 12)
            except Exception as e:
                self.logger.error("CR mark error: %s", e)

        if config.engine.dataout["cr"] and config.engine.dataout["pupil"]:
            try:
                pp = config.engine.dataout["pupil"]
                cr_list = config.engine.dataout["cr"]
                x_coords = [cr.center[0] for cr in cr_list]
                y_coords = [cr.center[1] for cr in cr_list]

                centroid_x = sum(x_coords) / len(cr_list)
                centroid_y = sum(y_coords) / len(cr_list)
                self.place_cross(source_rgb, (centroid_x, centroid_y), pink, 2, 12) # noqa: F405
                cv2.line(
                    source_rgb,
                    tuple_int(pp.center),
                    tuple_int((centroid_x, centroid_y)),
                    pink, # noqa: F405
                    3,
                )
            except Exception as e:
                self.logger.error("Pupil-CR line error: %s", e)


    def update(self, img: np.ndarray) -> None:
        """Update the GUI with the latest image and data."""
        self.print_cycle += 1

        if self.print_cycle == self.print_fps:
            source_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.draw(source_rgb)

            pupil_area = self.pupil_processor.source
            cv2.imshow(self.preview, source_rgb)
            cv2.imshow(self.pupil_bin, pupil_area)
            if config.engine.cr_processor is not None:
                cr_area = config.engine.cr_processor.source
                cv2.imshow(self.cr_bin, cr_area)
            cv2.waitKey(1)
            self.print_cycle = 0
