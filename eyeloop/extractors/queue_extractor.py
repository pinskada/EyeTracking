"""Queue extractor module for sending processed data via queues."""

import numpy as np

import eyeloop.config as config
from vr_core.utilities.logger_setup import setup_logger


class QueueExtractor:
    """Queue extractor for sending processed data via queues."""

    def __init__(self):
        self.logger = setup_logger(f"{config.arguments.side} QueueExtractor")
        self.response_queue = config.response_queue
        self.side = config.arguments.side
        self.eye_ready_signal = config.eye_ready_signal


    def activate(self):
        """Activate the QueueExtractor."""
        self.logger.info("Service activated.")


    def fetch(self, core):
        """Fetch processed data and send it via queues."""
        if config.importer != 0:

            # Signal to FrameProvider to fetch the next frame
            self.eye_ready_signal.set()
            self.logger.info("eye_ready_s set.")


            # Create message with tracking data
            tracking_data_message = {
                "type": "eye_data",
                "frame_id": config.current_frame_id,
                "data": core.dataout
            }

            # Send tracking data message via response queue
            self.response_queue.put(tracking_data_message)
            self.logger.info("response_queue: tracking_data sent.")

            if config.preview:

                # Create message with image preview data

                image_preview = config.graphical_user_interface.bin_P

                image_height = image_preview.shape[0]
                image_width = image_preview.shape[1]

                bit_map = np.packbits(image_preview.astype(np.uint8))

                preview_image_message = {
                    "type": "image_preview",
                    "frame_id": config.current_frame_id,
                    "height": image_height,
                    "width": image_width,
                    "bitmap": bit_map
                }

                # Send image preview message via response queue
                self.response_queue.put(preview_image_message)
                self.logger.info("response_queue: preview_image sent.")


    def __name__(self):
        """Return the name of the extractor."""
        return "QueueExtractor"


    #  pylint: disable=unused-argument
    def release(self, core):
        """Release resources held by the QueueExtractor."""
