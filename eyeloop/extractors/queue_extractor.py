import eyeloop.config as config
from eyeloop.utilities.encode_binary_float64_as_png import encode_binary_float64_as_png

class QueueExtractor:
    def __init__(self):
        pass

    def activate(self):
        return
    
    def fetch(self, core):
       
        if config.show_bin:
            try:
                png_image = encode_binary_float64_as_png(config.graphical_user_interface.bin_P)

                config.response_queue.put({"bin_image": png_image})
            except ValueError:
                pass

        else:
            try:

                config.response_queue.put({core.dataout})
            except ValueError:
                pass

    def release(self, core):
        pass