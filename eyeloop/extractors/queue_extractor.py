"""Queue extractor module for sending processed data via queues."""

import numpy as np

import eyeloop.config as config


class QueueExtractor:
    """Queue extractor for sending processed data via queues."""

    def __init__(self):
        self.response_queue = config.response_queue
        self.side = config.arguments.side


    def activate(self):
        """Activate the QueueExtractor."""
        print(f"[INFO] Extractor {self.side}: QueueExtractor activated.")


    def fetch(self, core):
        """Fetch processed data and send it via queues."""
        if config.importer != 0:

            # Signal to FrameProvider to fetch the next frame
            config.eye_ready_signal.set()

            # Create message with tracking data
            tracking_data_message = {
                "type": "eye_data",
                "frame_id": config.current_frame_id,
                "data": core.dataout
            }

            # Send tracking data message via response queue
            config.response_queue.put(tracking_data_message)

            if config.preview:

                # Create message with image preview data

                image_preview = config.graphical_user_interface.bin_P

                image_height = image_preview.shape[0]
                image_width = image_preview.shape[1]

                bit_map = np.packbits(image_preview.astype(np.uint8))

                tracking_data_message = {
                    "type": "image_preview",
                    "frame_id": config.current_frame_id,
                    "height": image_height,
                    "width": image_width,
                    "bitmap": bit_map
                }

                # Send image preview message via response queue
                config.response_queue.put(tracking_data_message)


    def __name__(self):
        """Return the name of the extractor."""
        return "QueueExtractor"


    #  pylint: disable=unused-argument
    def release(self, core):
        """Release resources held by the QueueExtractor."""
        print(f"[INFO] Extractor {self.side}: QueueExtractor released called, passing.")
