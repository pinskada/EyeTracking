import eyeloop.config as config
from eyeloop.utilities.encode_binary_float64_as_png import encode_binary_float64_as_png

class QueueExtractor:
    def __init__(self):
        self.response_queue = config.response_queue
        self.sync_queue = config.sync_queue

    def activate(self):
        print("[Extractor] QueueExtractor activated.")
    
    def fetch(self, core):
        try:
            self.sync_queue.put({"type": "ack", "frame_id": config.importer.frame_id})
        except ValueError as e:
            print(f"[ERROR] Error writing to sync queue: {e}")

        if config.preview:
            try:
                png_image = encode_binary_float64_as_png(config.graphical_user_interface.bin_P)

                config.response_queue.put({"bin_image": png_image})
            except ValueError as e:
                print(f"[ERROR] QueueExtractor.fetch() ValueError: {e}")
   
        try:
            config.response_queue.put({core.dataout})
        except ValueError as e:
            print(f"[ERROR] QueueExtractor.fetch() error: {e}")

    def __name__(self):
        return "QueueExtractor"

    def release(self):
        print("[Extractor] QueueExtractor released called, passing.")
        pass
    