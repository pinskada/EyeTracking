import eyeloop.config as config
from eyeloop.utilities.encode_binary_float64_as_png import encode_binary_float64_as_png
import time
class QueueExtractor:
    def __init__(self):
        self.response_queue = config.response_queue
        self.acknowledge_queue = config.acknowledge_queue
        self.side = config.arguments.side

    def activate(self):
        print(f"[INFO] Extractor {self.side}: QueueExtractor activated.")
    
    def fetch(self, core):
        if config.importer.current_frame_id != 0:
            
            if config.preview:
                try:
                    png_image = encode_binary_float64_as_png(config.graphical_user_interface.bin_P)

                    config.response_queue.put(png_image)
                except ValueError as e:
                    print(f"[ERROR] Extractor {self.side}: QueueExtractor.fetch() ValueError: {e}")
            
            try:
                message = {
                    "type": "payload",
                    "frame_id": config.importer.current_frame_id,
                    "data": core.dataout
                }
                #print(f"[INFO] Extractor {self.side}: Sending response with frame ID: {config.importer.current_frame_id}")
                config.response_queue.put(message)
            except ValueError as e:
                print(f"[ERROR] Extractor {self.side}: QueueExtractor.fetch() error: {e}")
            time.sleep(0.005)
            try:
                self.acknowledge_queue.put({"type": "ack", "frame_id": config.importer.current_frame_id})
                #print(f"[INFO] Extractor {self.side}: Acknowledged frame ID: {config.importer.current_frame_id}")
            except ValueError as e:
                print(f"[ERROR] Extractor {self.side}: Error writing to acknowledge queue: {e}")
            
    def __name__(self):
        return "QueueExtractor"

    def release(self, core):
        print("[INFO] Extractor {self.side}: QueueExtractor released called, passing.")
        pass
    