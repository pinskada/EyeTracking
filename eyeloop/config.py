# pylint: disable=invalid-name
"""Configuration module for EyeLoop eye-tracking system."""
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent

import numpy as np
from eyeloop.engine.engine import Engine
from eyeloop.guis.minimum.minimum_gui import GUI
from eyeloop.importers.importer import IMPORTER
from eyeloop.utilities.argument_parser import Arguments
from eyeloop.utilities.file_manager import File_Manager

blink = np.zeros(30, dtype=np.float64)

blink_i = 0

version = "0.35-beta"
importer: IMPORTER
eyeloop = 0
engine: Engine
arguments: Arguments
file_manager: File_Manager
graphical_user_interface: GUI

tracker_cmd_q: mp.Queue
tracker_response_q: mp.Queue

eye_ready_signal: MpEvent
tracker_shm_is_closed_s: MpEvent
tracker_running_s: MpEvent

preview: str = "none"
current_frame_id: int

use_gui: bool
