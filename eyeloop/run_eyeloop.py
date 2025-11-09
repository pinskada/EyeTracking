"""EyeLoop main runnable module."""

from pathlib import Path
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent

import eyeloop.config as config
from eyeloop.engine.engine import Engine
from eyeloop.extractors.queue_extractor import QueueExtractor
from eyeloop.guis.minimum.minimum_gui import GUI as Minimum_GUI
from eyeloop.utilities.argument_parser import Arguments
from eyeloop.utilities.file_manager import File_Manager
from eyeloop.importers.shared_memory_importer import Importer
from vr_core.utilities.logger_setup import setup_logger

EYELOOP_DIR = Path(__file__).parent
PROJECT_DIR = EYELOOP_DIR.parent


class EyeLoop:
    """
    EyeLoop is a Python 3-based eye-tracker tailored specifically
    to dynamic, closed-loop experiments on consumer-grade hardware.

    Lead developer: Simon Arvin
    Git: https://github.com/simonarvin/eyeloop
    """

    def __init__(
        self,
        args,
        command_queue: mp.Queue,
        response_queue: mp.Queue,
        eye_ready_signal: MpEvent,
        tracker_shm_is_closed_signal: MpEvent,
        tracker_running_signal: MpEvent,
        logger=None,
    ):
        """Initialize EyeLoop with the specified configuration."""

        self.logger = setup_logger("EyeLoop_Run")
        if (command_queue is None or
            response_queue is None or
            eye_ready_signal is None or
            tracker_shm_is_closed_signal is None
        ):
            print("(!) No queues provided, aborting.")
            return
        else:
            config.command_queue = command_queue
            config.response_queue = response_queue
            config.eye_ready_signal = eye_ready_signal
            config.tracker_shm_is_closed_signal = tracker_shm_is_closed_signal
            config.tracker_running_signal = tracker_running_signal

        config.arguments = Arguments(args)
        config.file_manager = File_Manager(
            output_root=config.arguments.output_dir,
            img_format=config.arguments.img_format
        )

        #self.logger.info("Service initialized.")

        self.run()


    def run(self):
        """Run EyeLoop with the specified configuration."""
        #try:
        #    config.blink = np.load(f"{EYELOOP_DIR}/blink_.npy")[0] * .8
        #except:
        #    print("\n(!) NO BLINK DETECTION. Run 'eyeloop --blink 1' to calibrate\n")


        config.graphical_user_interface = Minimum_GUI()

        config.engine = Engine(self)

        extractors = [QueueExtractor()]

        config.engine.load_extractors(extractors)

        self.run_importer()


    def run_importer(self):
        """Run the selected importer module."""
        try:
            #logger.info(f"Initiating tracking via Importer: {config.arguments.importer}")
            config.importer = Importer()
            config.importer.route()

            # exec(import_command, globals())

        except ImportError:
            self.logger.error("Invalid importer selected")
