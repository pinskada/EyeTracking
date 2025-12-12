# ruff: noqa: ERA001, ANN401


"""Queue extractor module for sending processed data via queues."""

from typing import Any

import numpy as np
from eyeloop import config

import vr_core.eye_tracker.tracker_types as tt
from vr_core.utilities.logger_setup import setup_logger


class QueueExtractor:
    """Queue extractor for sending processed data via queues."""

    def __init__(self) -> None:
        """Initialize the QueueExtractor."""
        self.logger = setup_logger(f"{config.arguments.side} QueueExtractor")
        self.tracker_response_q = config.tracker_response_q
        self.side = config.arguments.side
        self.eye_ready_signal = config.eye_ready_signal
        self.print_state = 0

    def activate(self) -> None:
        """Activate the QueueExtractor."""
        #self.logger.info("Service activated.")


    def fetch(self, core: Any) -> None:
        """Fetch processed data and send it via queues."""
        if config.importer != 0:

            # Signal to FrameProvider to fetch the next frame
            self.eye_ready_signal.set()
            #self.logger.info("eye_ready_s set.")

            pupil_data = core.dataout["pupil"]
            cr_data = core.dataout["cr"]

            tracker_data = tt.OneSideTrackerData(
                pupil=pupil_data,
                crs=cr_data,
            )

            # Create message with tracking data
            tracking_data_message = {
                "type": "eye_data",
                "frame_id": config.current_frame_id,
                "data": tracker_data,
            }

            # Send tracking data message via response queue
            self.tracker_response_q.put(tracking_data_message)
            #self.logger.info("Tracker data for %s eye sent with ID: %s.",
            #   self.side, config.current_frame_id)

            if config.preview != "none":

                # Create message with image preview data
                self.print_state += 1
                if config.preview == "pupil":
                    image_preview = config.engine.pupil_processor.source
                elif config.preview == "cr":
                    image_preview = config.engine.cr_processor.source
                else:
                    self.logger.error("Unknown preview type: %s", config.preview)
                    return

                # mean_image = np.mean(image_preview)
                # self.logger.info("Mean image value: %s", mean_image)
                # if config.preview and self.print_state % 50 == 0:
                #     cv2.imwrite(f"/tmp/preview_{self.print_state}.png", image_preview)

                bin_img = (image_preview > 0).astype(np.uint8)
                image_height, image_width = bin_img.shape[:2]
                bit_map = np.packbits(bin_img, axis=None, bitorder="big")

                preview_image_message = {
                    "type": "image_preview",
                    "frame_id": config.current_frame_id,
                    "height": image_height,
                    "width": image_width,
                    "bitmap": bit_map.tobytes(),
                }

                # Send image preview message via response queue
                self.tracker_response_q.put(preview_image_message)
                #self.logger.info("Tracker preview for %s eye sent with ID: %s.",
                #   self.side, config.current_frame_id)


    def __name__(self) -> None:
        """Return the name of the extractor."""
        return "QueueExtractor"


    #  pylint: disable=unused-argument
    def release(self, core: Any) -> None:
        """Release resources held by the QueueExtractor."""
