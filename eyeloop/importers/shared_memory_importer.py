# ruff: noqa: ERA001, TRY400

"""Shared Memory Importer for Eyeloop Module."""

import queue
import time
from multiprocessing.shared_memory import SharedMemory
from threading import Event

import numpy as np
from eyeloop import config

from vr_core.utilities.logger_setup import setup_logger


class Importer:
    """Shared Memory Importer for Eyeloop Module."""

    def __init__(self) -> None:
        """Initialize the Shared Memory Importer."""
        self.logger = setup_logger("EyeLoop_Importer")
        self.scale = config.arguments.scale

        self.shared_memory_name = config.arguments.sharedmem
        self.side = config.arguments.side
        self.frame_shape: tuple[int, int]
        self.frame_dtype: np.dtype
        self.frame = None
        self.print_status = False

        self.tracker_cmd_q = config.tracker_cmd_q
        self.tracker_shm_is_closed_s = config.tracker_shm_is_closed_s
        self.tracker_running_s = config.tracker_running_s

        self.tracker_shm_is_closed_s.set()
        # self.logger.info("<%s> tracker_shm_is_closed_s set.", self.side)

        self.set_stop_event = Event()
        self.new_frame_event = Event()

        self.shm: SharedMemory

        self.time_1: float = 0.0
        self.time_total: float = 0.0
        self.time_to_import_total: float = 0.0
        self.time_cycle: int = 0
        self.time_cycle2: int = 0
        self.print_cycle: int = 100


    def route(self) -> None:
        """Start routing frames from shared memory to the engine."""
        loop_number = 0
        self.tracker_running_s.set()
        # self.logger.info("<%s> tracker_running_s is set.", self.side)
        while True:
            loop_number += 1
            self._load_cmd_queue_message()
            if (
                # SHM has been set and new frame arrived
                not self.tracker_shm_is_closed_s.is_set() and
                self.new_frame_event.is_set()
            ):
                break
            # self.logger.info("<%s> %d. loop proceeded without new frame or tracker_shm.",
            #     self.side, loop_number)

        # self.logger.info("<%s> First frame received, proceeding to arm engine.", self.side)
        self._first_frame()
        # self.logger.info("<%s> Starting routing.", self.side)
        self._proceed()


    def _load_cmd_queue_message(self, timeout: float = 0.005) -> bool:
        """Load a message from the command queue."""
        try:
            msg = self.tracker_cmd_q.get(timeout=timeout)
        except queue.Empty:
            # self.logger.info("tracker_cmd_q is empty.")
            return False

        if msg.get("type") == "shm_connect":
            # self.logger.info("<%s> SHM connect command received.", self.side)
            self._connect_shm(msg)
        elif msg.get("type") == "shm_detach":
            # self.logger.info("<%s> SHM detach command received.", self.side)
            self._close_shm()
        elif msg.get("type") == "close":
            # self.logger.info("<%s> Close command received.", self.side)
            self.set_stop_event.set()
            #self.logger.info("set_stop_event set.")
            config.engine.release()
        elif msg.get("type") == "config":
            # self.logger.info("<%s> Config command received.", self.side)
            self._configure(msg)
        elif msg.get("type") == "frame_id":
            frame_id = msg.get("value")
            # self.logger.info("<%s> Received ID: %d", self.side, frame_id)
            config.current_frame_id = frame_id
            self.new_frame_event.set()
        else:
            self.logger.warning("<%s> Unknown command: %s", self.side, msg.get("type"))

        return True


    def _first_frame(self) -> None:
        """Process the first frame to arm the engine."""
        try:
            frame = self._get_frame()
        except Exception as e:  # pylint: disable=broad-except  # noqa: BLE001
            self.logger.error("<%s> Error in first frame processing: %s", self.side, e)
            return

        # self.logger.info("<%s> Arming the engine...", self.side)
        config.engine.arm(
            height=self.frame_shape[0],
            width=self.frame_shape[1],
            image=frame,
        )


    def _get_frame(self) -> np.ndarray:
        """Get a frame from the Shared Memory."""
        return np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shm.buf).copy()


    def _proceed(self) -> None:
        """Route frames from shared memory to the engine."""
        # self.logger.info("<%s> Entering _proceed loop.", self.side)
        while not self.set_stop_event.is_set():

            # Check for new commands
            if self._load_cmd_queue_message():
                self.time_cycle2 += 1
                if self.time_cycle2 > 10:  # noqa: PLR2004
                    self.time_to_import_total += time.perf_counter_ns() / 1e9 - self.time_1
                if self.time_cycle2 == self.print_cycle:
                    # self.logger.info(
                    #     "Import time: %.3f ms;",
                    #     (self.time_to_import_total / self.print_cycle) * 1000)
                    self.time_cycle2 = 0
                    self.time_to_import_total = 0.0


            # Skip processing if shared memory is closed
            if self.tracker_shm_is_closed_s.is_set():
                continue

            # print status every 50 frames
            if config.current_frame_id % 50 == 0 and self.print_status:
                self.logger.info("<%s> Current frame ID: %d", self.side, config.current_frame_id)

            # Skip if no new frame is yet available
            if self.new_frame_event.is_set():
                # Get frame from shared memory and iterate
                frame = self._get_frame()

                if self.time_cycle != 0:
                    self.time_total += time.perf_counter_ns() / 1e9 - self.time_1

                self.time_cycle += 1
                if self.time_cycle == self.print_cycle:
                    # self.logger.info(
                    #     "Import time: %.3f ms;",
                    #     (self.time_total / self.print_cycle) * 1000)
                    self.time_cycle = 0
                    self.time_total = 0.0

                config.engine.track(frame)

                self.time_1 = time.perf_counter_ns() / 1e9
                self.new_frame_event.clear()
                #self.logger.info("new_frame_event cleared.")


    def _configure(self, msg: dict,  # noqa: C901, PLR0912
    ) -> None:
        """Configure the run-time parameters based on the received message."""
        try:
            if  msg.get("param") == "threshold_pupil":
                thr = msg.get("value")
                config.engine.pupil_processor.binarythreshold = thr
            elif  msg.get("param") == "threshold_cr":
                thr = msg.get("value")
                config.engine.cr_processor.binarythreshold = thr

            elif msg.get("param") == "blur_size_pupil":
                blur = msg.get("value")
                if blur % 2 == 0:
                    blur += 1
                config.engine.pupil_processor.blur = (blur, blur)

            elif msg.get("param") == "blur_size_cr":
                blur = msg.get("value")
                if blur % 2 == 0:
                    blur += 1
                config.engine.cr_processor.blur = (blur, blur)

            elif msg.get("param") == "mask_radius_cr":
                config.engine.cr_processor.mask_radius = msg.get("value")
                # self.logger.info("<%s> mask_radius_cr set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "min_radius_pupil":
                config.engine.pupil_processor.min_radius = msg.get("value")
                config.engine.pupil_processor.distance_transform.min_radius = msg.get("value")
                # self.logger.info("<%s> minR set to %d", self.side, msg.get("value"))
            elif msg.get("param") == "min_radius_cr":
                config.engine.cr_processor.min_radius = msg.get("value")
                config.engine.cr_processor.distance_transform.min_radius = msg.get("value")
                # self.logger.info("<%s> minR set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "max_radius_pupil":
                config.engine.pupil_processor.max_radius = msg.get("value")
                config.engine.pupil_processor.distance_transform.max_radius = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))
            elif msg.get("param") == "max_radius_cr":
                config.engine.cr_processor.max_radius = msg.get("value")
                config.engine.cr_processor.distance_transform.max_radius = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "min_circularity_pupil":
                config.engine.pupil_processor.distance_transform.circularity_min = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))
            elif msg.get("param") == "max_circularity_pupil":
                config.engine.pupil_processor.distance_transform.circularity_max = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "min_circularity_cr":
                config.engine.cr_processor.distance_transform.circularity_min = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))
            elif msg.get("param") == "max_circularity_cr":
                config.engine.cr_processor.distance_transform.circularity_max = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "max_aspect_ratio_pupil":
                config.engine.pupil_processor.distance_transform.aspect_ratio_max = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))
            elif msg.get("param") == "max_aspect_ratio_cr":
                config.engine.cr_processor.distance_transform.aspect_ratio_max = msg.get("value")
                # self.logger.info("<%s> maxR set to %d", self.side, msg.get("value"))

            elif msg.get("param") == "preview":
                config.preview = msg.get("value")
                # self.logger.info("<%s> Preview set to %d", self.side, msg.get("value"))

            else:
                self.logger.warning("<%s> Unknown configuration parameter: %s",
                    self.side, msg.get("param"))

        except (TypeError, ValueError, KeyError) as e:
            self.logger.warning("<%s> Failed to apply config: %s", self.side, e)


    def _connect_shm(self, msg: dict) -> None:
        """Set up the shared memory segment."""
        self.frame_shape = tuple(msg["frame_shape"])
        self.frame_dtype = np.dtype(msg["frame_dtype"])
        self.shm = SharedMemory(name=self.shared_memory_name)
        self.tracker_shm_is_closed_s.clear()
        # self.logger.info("<%s> tracker_shm_is_closed_s is cleared.", self.side)


    def _close_shm(self) -> None:
        """Close the shared memory segment."""
        if not self.tracker_shm_is_closed_s.is_set():
            try:
                self.shm.close()
                self.tracker_shm_is_closed_s.set()
                self.logger.info("<%s> tracker_shm_is_closed_s is set.", self.side)
            except (FileNotFoundError, PermissionError, OSError, BufferError) as e:
                self.logger.error("<%s> Error closing shared memory: %s", self.side, e)


    def release(self) -> None:
        """Release the importer and shared memory."""

        #self._close_shm()
        #cv2.destroyAllWindows()
