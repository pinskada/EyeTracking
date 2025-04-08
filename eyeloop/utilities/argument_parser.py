import argparse
from pathlib import Path

EYELOOP_DIR = Path(__file__).parent.parent
PROJECT_DIR = EYELOOP_DIR.parent


class Arguments:
    """
    Parses all command-line arguments and config.pupt parameters.
    """

    def __init__(self, args) -> None:
        self.config = None
        self.markers = None
        self.video = None
        self.output_dir = None
        self.importer = None
        self.scale = None
        self.tracking = None
        self.model = None
        self.side = None
        self.min_radius_threshold = None
        self.max_radius_threshold = None
        self.search_step = None
        self.auto_search = None
    

        self.parsed_args = self.parse_args(args)
        self.build_config(parsed_args=self.parsed_args)

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser(description='Help list')
        parser.add_argument("-v", "--video", default="0", type=str,
                            help="Input a video sequence for offline processing.")

        parser.add_argument("-o", "--output_dir", default=str(PROJECT_DIR.joinpath("data").absolute()), type=str,
                            help="Specify output destination.")
        parser.add_argument("-c", "--config", default="0", type=str, help="Input a .pupt config file (preset).")
        parser.add_argument("-i", "--importer", default="cv", type=str,
                            help="Set import route of stream (cv, vimba, ...)")
        parser.add_argument("-sc", "--scale", default=1, type=float, help="Scale the stream (default: 1; 0-1)")
        parser.add_argument("-m", "--model", default="ellipsoid", type=str,
                            help="Set pupil model type (circular; ellipsoid = default).")
        parser.add_argument("-ma", "--markers", default=0, type=int,
                            help="Enable/disable artifact removing markers (0: disable/default; 1: enable)")
        parser.add_argument("-tr", "--tracking", default=1, type=int,
                            help="Enable/disable tracking (1/enabled: default).")

        parser.add_argument("-ex", "--extractors", default="", type=str,
                            help="Set file-path of extractor Python file. p = start file prompt.")

        parser.add_argument("-imgf", "--img_format", default="frame_$.jpg", type=str,
                            help="Set img format for import (default: frame_$.jpg where $ = 1, 2,...)")

        parser.add_argument("-sv", "--save", default=1, type=int,
                            help="Save video feed or not (yes/no, 1/0; default = 1)")

        parser.add_argument("-rt", "--rotation", default=0, type=int,
                            help="Enable online rotation (yes/no, 1/0; default = 0)")

        parser.add_argument("-fps", "--framerate", default=1, type=float,
                            help="How often to update preview window  (default = 1/second)")

        parser.add_argument("-cl", "--clear", default=0, type=float,
                            help="Clear parameters (yes/no, 1/0) - default = 0")

        parser.add_argument("-p", "--params", default="", type=str,
                            help="Load pupil/cr parameter file (.npy)")

        parser.add_argument("-b", "--blink", default="", type=str,
                            help="Load blink calibration file (.npy)")
        
        parser.add_argument("-s", "--side", default="B", type=str,
                            help="Chooses what side of the image to work with")
        
        parser.add_argument("-thr", "--min_radius_threshold", default = 5, type=int,
                            help="Minimal radius threshold for pupil fitting")
        
        parser.add_argument("-mthr", "--max_radius_threshold", default = 20, type=int,
                            help="Maximum radius threshold for pupil fitting")
        
        parser.add_argument("-srs", "--search_step", default=20, type=int,                            
                            help="Step with which the pupil search patterns increases")
        
        parser.add_argument("-as", "--auto_search", default=1, type=int,
                            help="Automatic search for pupil (yes/no, 1/0) - default = 1")
        
        return parser.parse_args(args)

    def build_config(self, parsed_args):
        self.config = parsed_args.config

        if self.config != "0":  # config file was set.
            self.parse_config(self.config)

        self.markers = parsed_args.markers
        self.video = Path(parsed_args.video.strip("\'\"")).absolute()  # Handle quotes used in argument
        self.output_dir = Path(parsed_args.output_dir.strip("\'\"")).absolute()
        self.importer = parsed_args.importer.lower()
        self.scale = parsed_args.scale
        self.tracking = parsed_args.tracking
        self.model = parsed_args.model.lower()
        self.extractors = parsed_args.extractors
        self.img_format = parsed_args.img_format
        self.save = parsed_args.save
        self.rotation = parsed_args.rotation
        self.fps = parsed_args.framerate
        self.clear = parsed_args.clear
        self.params = parsed_args.params
        self.blinkcalibration = parsed_args.blink
        self.side = parsed_args.side
        self.min_radius_threshold = parsed_args.min_radius_threshold
        self.max_radius_threshold = parsed_args.max_radius_threshold
        self.search_step = parsed_args.search_step
        self.auto_search = parsed_args.auto_search
        
        #self.blink = parsed_args.blink

    def parse_config(self, config: str) -> None:
        with open(config, "r") as content:
            print("Loading config preset: ", config)
            for line in content:
                split = line.split("=")
                parameter = split[0]
                parameter = split[1].rstrip("\n").split("\"")

                if len(parameter) != 1:
                    parameter = parameter[1]
                else:
                    parameter = parameter[0]

                if parameter == "video":
                    print("Video preset: ", parameter)
                    self.video = parameter
                elif parameter == "dest":
                    print("Destination preset: ", parameter)
                    self.output_dir = Path(parameter).absolute()

                elif parameter == "import":
                    print("Importer preset: ", parameter)
                    self.importer = parameter
                elif parameter == "model":
                    print("Model preset: ", parameter)
                    self.model = parameter
                elif parameter == "markers":
                    print("Markers preset: ", parameter)
                    self.markers = parameter
                elif parameter == "extractors":
                    print("Extractors preset: ", parameter)
                    self.extractors = parameter
                elif parameter == "img_format":
                    print("img_format preset: ", parameter)
                    self.img_format = parameter
                elif parameter == "save":
                    print("save preset: ", parameter)
                    self.save = parameter
                elif parameter == "rotation":
                    print("rotation preset: ", parameter)
                    self.rotation = parameter
                elif parameter == "framerate":
                    print("framerate preset: ", parameter)
                    self.fps = parameter
                elif parameter == "side":
                    print("working with side: ", parameter)
                    self.side = parameter
                elif parameter == "min_radius_threshold":
                    print("minimal pupil radius threshold: ", parameter)
                    self.min_radius_threshold = parameter
                elif parameter == "max_radius_threshold":
                    print("maximum radius threshold: ", parameter)
                    self.max_radius_threshold = parameter
                elif parameter == "search_step":
                    print("search step: ", parameter)
                    self.search_step = parameter
                elif parameter == "auto_search":
                    print("auto_search: ", parameter)
                    self.auto_search = parameter
                

            print("")
