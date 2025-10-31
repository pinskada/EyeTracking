"""Shared Memory Importer for Eyeloop Module."""

from multiprocessing.shared_memory import SharedMemory
from threading import Event

import numpy as np

import eyeloop.config as config
from vr_core.utilities.logger_setup import setup_logger


class Importer():
    """Shared Memory Importer for Eyeloop Module."""

    def __init__(self) -> None:
        """Initialize the Shared Memory Importer."""

        self.logger = setup_logger(f"{config.arguments.side} EyeLoop_Importer")
        self.scale = config.arguments.scale

        # Wait for init command via command_queue
        self.shared_memory_name = config.arguments.sharedmem
        self.side = config.arguments.side
        self.frame_shape: tuple[int, int]
        self.frame_dtype: np.dtype
        self.frame = None
        self.print_status = False

        self.command_queue = config.command_queue
        self.tracker_shm_is_closed_signal = config.tracker_shm_is_closed_signal

        self.tracker_shm_is_closed_signal.set()
        self.logger.info("tracker_shm_is_closed_s set.")

        self.set_stop_event = Event()
        self.new_frame_event = Event()

        self.shm: SharedMemory


    def route(self) -> None:
        """Start routing frames from shared memory to the engine."""

        self._load_cmd_queue_message(timeout=10.0)

        self._first_frame()
        self.logger.info("Starting routing.")
        self._proceed()


    def _load_cmd_queue_message(self, timeout: float = 5.0) -> None:
        """Loads a message from the command queue."""

        try:
            if not (msg := self.command_queue.get(timeout=timeout)):
                self.logger.error("No message received from command queue.")
                return

            if self.tracker_shm_is_closed_signal.is_set() and msg.get("type") != "shm_connect":
                self.logger.warning("First message must be 'shm_connect'.")
                return

            if msg.get("type") == "shm_connect":
                self._connect_shm(msg)
            elif msg.get("type") == "shm_detach":
                self._close_shm()
            elif msg.get("type") == "close":
                self.set_stop_event.set()
                self.logger.info("set_stop_event set.")
                config.engine.release()
            elif msg.get("type") == "config":
                self._configure(msg)
            elif msg.get("type") == "frame_id":
                config.current_frame_id = msg.get("frame_id")
                self.new_frame_event.set()
            else:
                self.logger.warning("Unknown command: %s", msg.get('type'))
        except Exception:  # pylint: disable=broad-except
            pass


    def _first_frame(self) -> None:
        """Process the first frame to arm the engine."""

        try:
            frame = self._get_frame()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error in first frame processing: %s", e)
            return

        self.logger.info("Arming the engine...")
        config.engine.arm(
            height=self.frame_shape[0],
            width=self.frame_shape[1],
            image=frame
        )


    def _get_frame(self) -> np.ndarray:
        """Get a frame from the Shared Memory."""

        return np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shm.buf).copy()


    def _proceed(self) -> None:
        """Main loop to route frames from shared memory to the engine."""

        self.logger.info("Entering _proceed loop.")
        while not self.set_stop_event.is_set():

            # Check for new commands
            self._load_cmd_queue_message()

            # Skip processing if shared memory is closed
            if self.tracker_shm_is_closed_signal.is_set():
                continue

            # print status every 50 frames
            if config.current_frame_id % 50 == 0 and self.print_status:
                self.logger.info("Current frame ID: %d", config.current_frame_id)

            # Skip if no new frame is yet available
            if self.new_frame_event.is_set():
                # Get frame from shared memory and iterate
                frame = self._get_frame()
                config.engine.iterate(frame)
                self.new_frame_event.clear()
                self.logger.info("new_frame_event cleared.")


    def _configure(self, msg):
        """Configure the run-time parameters based on the received message."""

        try:
            if  msg.get("param") == "threshold":
                value = msg.get("value")
                config.graphical_user_interface.pupil_processor.binarythreshold = value
                self.logger.info("Threshold decreased to %d",
                    config.graphical_user_interface.pupil_processor.binarythreshold
                )

            elif msg.get("param") == "blur":
                blur = config.graphical_user_interface.pupil_processor.blur
                if blur[0] >= 3 and blur[1] >= 3:
                    value = msg.get("value")
                    config.graphical_user_interface.pupil_processor.blur = (value, value)
                    self.logger.info("Blur decreased to %s.",
                        config.graphical_user_interface.pupil_processor.blur
                    )
                else:
                    self.logger.warning("Incorrect blur values: %s.", blur)

            elif msg.get("param") == "auto_search":
                config.arguments.auto_search = msg.get("value")
                self.logger.info("auto_search set to %d", msg.get("value"))

            elif msg.get("param") == "minThrRad":
                config.graphical_user_interface.pupil_processor.min_radius = msg.get("value")
                self.logger.info("minR set to %d", msg.get("value"))

            elif msg.get("param") == "maxThrRad":
                config.graphical_user_interface.pupil_processor.max_radius = msg.get("value")
                self.logger.info("maxR set to %d", msg.get("value"))

            elif msg.get("param") == "search_step":
                config.arguments.search_step = msg.get("value")
                self.logger.info("search step set to %d", msg.get("value"))
            elif msg.get("param") == "preview":
                config.preview = msg.get("value")
                self.logger.info("Preview set to %d", msg.get("value"))
            else:
                self.logger.warning("Unknown configuration parameter: %s", msg.get("param"))

        except (TypeError, ValueError, KeyError) as e:
            self.logger.warning("Failed to apply config: %s", e)


    def _connect_shm(self, msg):
        """Setup the shared memory segment."""

        self.frame_shape = tuple(msg["frame_shape"])
        self.frame_dtype = np.dtype(msg["frame_dtype"])
        self.shm = SharedMemory(name=self.shared_memory_name)
        self.tracker_shm_is_closed_signal.clear()
        self.logger.info("tracker_shm_is_closed_s is cleared.")


    def _close_shm(self):
        """Close the shared memory segment."""
        try:
            self.shm.close()
            self.tracker_shm_is_closed_signal.set()
            self.logger.info("tracker_shm_is_closed_s is set.")
        except (FileNotFoundError, PermissionError, OSError, BufferError) as e:
            self.logger.error("Error closing shared memory: %s", e)


    def release(self) -> None:
        """Release the importer and shared memory."""

        self._close_shm()
        #cv2.destroyAllWindows()
