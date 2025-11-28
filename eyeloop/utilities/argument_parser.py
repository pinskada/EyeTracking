# ruff: noqa: T201

"""Argument parser for Eyeloop module."""

import argparse
from pathlib import Path

EYELOOP_DIR = Path(__file__).parent.parent
PROJECT_DIR = EYELOOP_DIR.parent


class Arguments:
    """Parses all command-line arguments and config.pupt parameters."""

    def __init__(self, args: list[str]) -> None:
        """Initialize and parse arguments."""
        self.config = None
        self.markers = None
        self.video = None
        self.output_dir: str
        self.importer = None
        self.scale = None
        self.tracking = None
        self.model = None
        self.side: str
        self.min_radius_threshold = None
        self.max_radius_threshold = None
        self.search_step = None
        self.auto_search = None
        self.tracker_fps = None
        self.sharedmem = None
        self.extractors = None
        self.img_format: str
        self.save = None
        self.rotation = None
        self.fps = None
        self.use_gui: int
        self.eng_profiling: str
        self.proc_profiling: str

        self.parsed_args = self.parse_args(args)
        self.build_config(parsed_args=self.parsed_args)

    @staticmethod
    def parse_args(args: list[str]) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Help list")
        parser.add_argument("-v", "--video", default="0", type=str,
                            help="Input a video sequence for offline processing.")

        parser.add_argument("-o", "--output_dir",
                            default=str(PROJECT_DIR.joinpath("data").absolute()),
                            type=str,
                            help="Specify output destination.")

        parser.add_argument("-c", "--config",
                            default="0", type=str,
                            help="Input a .pupt config file (preset).")

        parser.add_argument("-i", "--importer", default="cv", type=str,
                            help="Set import route of stream (cv, vimba, ...)")

        parser.add_argument("-sc", "--scale",
                            default=1, type=float,
                            help="Scale the stream (default: 1; 0-1)")

        parser.add_argument("-m", "--model", default="fast_elliptical", type=str,
                            help="Set pupil model type (circular; elliprtical; fast_elliptical = default).")

        parser.add_argument("-ma", "--markers", default=0, type=int,
                            help="Enable/disable artifact removing markers "
                            "(0: disable/default; 1: enable)")

        parser.add_argument("-tr", "--tracking", default=1, type=int,
                            help="Enable/disable tracking (1/enabled: default).")

        parser.add_argument("-ex", "--extractors", default="", type=str,
                            help="Set file-path of extractor Python file. p = start file prompt.")

        parser.add_argument("-imgf", "--img_format", default="frame_$.jpg", type=str,
                            help="Set img format for import "
                            "(default: frame_$.jpg where $ = 1, 2,...)")

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

        parser.add_argument("-trf", "--tracker_fps", default=1000, type=int,
                            help="Refresh rate cap for the tracker (default = 1000)")

        parser.add_argument("-shm", "--sharedmem", default="", type=str,
                            help="Name of the shared memory segment "
                            "for inter-process communication.")
        parser.add_argument("-ug", "--use_gui", default="0", type=int,
                            help="Use graphical user interface (yes/no, 1/0; default = 0)")
        parser.add_argument("-eng", "--eng_profiling", default="none", type=str,
                            help="Set profiling output file (both, left, right, none = default).")
        parser.add_argument("-proc", "--proc_profiling", default="none", type=str,
                            help="Set profiling output file (both, left, right, none = default).")

        return parser.parse_args(args)

    def build_config(self, parsed_args: argparse.Namespace) -> None:
        """Build configuration from parsed arguments."""
        self.config = parsed_args.config

        if self.config != "0":  # config file was set.
            self.parse_config(self.config)

        self.markers = parsed_args.markers
        # Handle quotes used in arguments
        self.video = Path(parsed_args.video.strip("'\"")).absolute()
        self.output_dir = Path(parsed_args.output_dir.strip("'\"")).absolute()
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
        self.tracker_fps = parsed_args.tracker_fps
        self.sharedmem = parsed_args.sharedmem
        self.use_gui = parsed_args.use_gui
        self.eng_profiling = parsed_args.eng_profiling
        self.proc_profiling = parsed_args.proc_profiling

    def parse_config(self, config: str) -> None:  # noqa: C901, PLR0912, PLR0915
        """Parse a .pupt config file and sets parameters accordingly."""
        with open(config) as content:  # noqa: PTH123
            print("Loading config preset: ", config)
            for line in content:
                split = line.split("=")
                parameter = split[0]
                parameter = split[1].rstrip("\n").split('"')

                parameter = parameter[1] if len(parameter) != 1 else parameter[0]

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
                elif parameter == "tracker_fps":
                    print("tracker_fps: ", parameter)
                    self.tracker_fps = parameter
                elif parameter == "sharedmem":
                    print("sharedmem: ", parameter)
                    self.sharedmem = parameter
                elif parameter == "use_gui":
                    print("use_gui: ", parameter)
                    self.use_gui = parameter
                elif parameter == "eng_profiling":
                    print("eng_profiling: ", parameter)
                    self.eng_profiling = parameter
                elif parameter == "proc_profiling":
                    print("proc_profiling: ", parameter)
                    self.proc_profiling = parameter

            print()
