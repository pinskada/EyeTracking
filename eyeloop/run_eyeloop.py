import importlib
import logging
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

"""
current_file = os.path.abspath(__file__)
repo_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", ".."))

if repo_root not in sys.path:
    print(f"[DEBUG] Adding repo root to sys.path: {repo_root}")
    sys.path.insert(0, repo_root)
"""
eyeloop_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if eyeloop_root not in sys.path:
    sys.path.insert(0, eyeloop_root)
from multiprocessing import Queue

import eyeloop.config as config
from eyeloop.engine.engine import Engine
from eyeloop.extractors.DAQ import DAQ_extractor
from eyeloop.extractors.frametimer import FPS_extractor
from eyeloop.extractors.queue_extractor import QueueExtractor

from eyeloop.utilities.argument_parser import Arguments
from eyeloop.utilities.file_manager import File_Manager
from eyeloop.utilities.format_print import welcome
from eyeloop.utilities.shared_logging import setup_logging
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Queue

EYELOOP_DIR = Path(__file__).parent
PROJECT_DIR = EYELOOP_DIR.parent

logger = logging.getLogger(__name__)


class EyeLoop:
    """
    EyeLoop is a Python 3-based eye-tracker tailored specifically to dynamic, closed-loop experiments on consumer-grade hardware.
    Lead developer: Simon Arvin
    Git: https://github.com/simonarvin/eyeloop
    """

    def __init__(self, args, logger=None, command_queue=None, response_queue=None, sync_queue=None, acknowledge_queue=None):

        #welcome("Server")
        if command_queue is None or response_queue is None or sync_queue is None or acknowledge_queue is None:
            config.command_queue = Queue()
            config.response_queue = Queue()
            config.sync_queue = Queue()
            config.acknowledge_queue = Queue()
            print("(!) No queues provided. Creating empty Queues.")
        else:
            config.command_queue = command_queue
            config.response_queue = response_queue
            config.sync_queue = sync_queue
            config.acknowledge_queue = acknowledge_queue

        config.arguments = Arguments(args)
        config.file_manager = File_Manager(output_root=config.arguments.output_dir, img_format = config.arguments.img_format)
        if logger is None:
            logger, logger_filename = setup_logging(log_dir=config.file_manager.new_folderpath, module_name="run_eyeloop")

        #if config.arguments.blink == 0:
        self.run()
        #else:
        #    self.test_blink()

    def test_blink(self):
        from eyeloop.guis.blink_test import GUI
        config.graphical_user_interface = GUI()
        config.engine = Engine(self)
        self.run_importer()

    def run(self):
        #try:
        #    config.blink = np.load(f"{EYELOOP_DIR}/blink_.npy")[0] * .8
        #except:
        #    print("\n(!) NO BLINK DETECTION. Run 'eyeloop --blink 1' to calibrate\n")


        from eyeloop.guis.minimum.minimum_gui import GUI
        config.graphical_user_interface = GUI()

        config.engine = Engine(self)

        #fps_counter = FPS_extractor()
        #data_acquisition = DAQ_extractor(config.file_manager.new_folderpath)

        file_path = config.arguments.extractors

        if file_path == "p":
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()

        #extractors_add = []

        if file_path != "":
            try:
                #logger.info(f"including {file_path}")
                sys.path.append(os.path.dirname(file_path))
                module_import = os.path.basename(file_path).split(".")[0]

                extractor_module = importlib.import_module(f"eyeloop.extractors.{module_import}")
                extractors_add = extractor_module.extractors_add

            except Exception as e:
                print(f"extractors not included, {e}")

        #extractors_base = [fps_counter, data_acquisition]
        extractors = [QueueExtractor()]

        config.engine.load_extractors(extractors)

        self.run_importer()

    def run_importer(self):
        try:
            #logger.info(f"Initiating tracking via Importer: {config.arguments.importer}")
            importer_module = importlib.import_module(f"eyeloop.importers.{config.arguments.importer}")
            config.importer = importer_module.Importer()
            config.importer.route()

            # exec(import_command, globals())

        except ImportError:
            print("Invalid importer selected")


def main():
    EyeLoop(sys.argv[1:], logger=None, command_queue=None, response_queue=None, sync_queue=None, acknowledge_queue=None)


if __name__ == '__main__':

    main()
