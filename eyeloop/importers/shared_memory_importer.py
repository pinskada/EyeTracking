"""Shared Memory Importer for Eyeloop Module."""

from multiprocessing.shared_memory import SharedMemory
from threading import Event

import numpy as np

import eyeloop.config as config


class Importer():
    """Shared Memory Importer for Eyeloop Module."""

    def __init__(self) -> None:
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

        self.set_stop_event = Event()
        self.new_frame_event = Event()

        self.shm: SharedMemory


    def route(self) -> None:
        """Start routing frames from shared memory to the engine."""

        self._load_cmd_queue_message(timeout=10.0)

        self._first_frame()
        print(f"[INFO] Importer {self.side}: Starting routing...\n")
        self._proceed()


    def _load_cmd_queue_message(self, timeout: float = 5.0) -> None:
        """Loads a message from the command queue."""

        try:
            if not (msg := self.command_queue.get(timeout=timeout)):
                print(f"[ERROR] Importer {self.side}: No message received from command queue.")
                return

            if self.tracker_shm_is_closed_signal.is_set() and msg.get("type") != "shm_connect":
                print(f"[WARN] Importer {self.side}: First message must be 'shm_connect'.")
                return

            if msg.get("type") == "shm_connect":
                self._connect_shm(msg)
            elif msg.get("type") == "shm_detach":
                self._close_shm()
            elif msg.get("type") == "close":
                print(f"[INFO] Importer {self.side}: Closing shared memory...")
                self.set_stop_event.set()
                config.engine.release()
            elif msg.get("type") == "config":
                self._configure(msg)
            elif msg.get("type") == "frame_id":
                config.current_frame_id = msg.get("frame_id")
                self.new_frame_event.set()
            else:
                print(f"[INFO] Importer {self.side}: Unknown command: {msg.get('type')}")
        except Exception:  # pylint: disable=broad-except
            pass


    def _first_frame(self) -> None:
        """Process the first frame to arm the engine."""

        try:
            frame = self._get_frame()
        except Exception as e:  # pylint: disable=broad-except
            print(f"[ERROR] Importer {self.side}: Error in first frame processing: {e}")
            return

        print("Arming the engine...")
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

        while not self.set_stop_event.is_set():

            # Check for new commands
            self._load_cmd_queue_message()

            # Skip processing if shared memory is closed
            if self.tracker_shm_is_closed_signal.is_set():
                continue

            # print status every 50 frames
            if config.current_frame_id % 50 == 0 and self.print_status:
                print(f"[INFO] Importer {self.side}: Current frame ID: {config.current_frame_id}\n")

            # Skip if no new frame is yet available
            if self.new_frame_event.is_set():
                # Get frame from shared memory and iterate
                frame = self._get_frame()
                config.engine.iterate(frame)
                self.new_frame_event.clear()


    def _configure(self, msg):
        """Configure the run-time parameters based on the received message."""

        try:
            if  msg.get("param") == "threshold":
                value = msg.get("value")
                config.graphical_user_interface.pupil_processor.binarythreshold = value
                print(f"[INFO] Importer {self.side}: Threshold decreased to "
                    f"{config.graphical_user_interface.pupil_processor.binarythreshold}."
                )

            elif msg.get("param") == "blur":
                blur = config.graphical_user_interface.pupil_processor.blur
                if blur[0] >= 3 and blur[1] >= 3:
                    value = msg.get("value")
                    config.graphical_user_interface.pupil_processor.blur = (value, value)
                    print(f"[INFO] Importer {self.side}: "
                        f"Blur decreased to {config.graphical_user_interface.pupil_processor.blur}."
                    )
                else:
                    print(f"[INFO] Importer {self.side}: Minimum blur reached.")

            elif msg.get("param") == "auto_search":
                config.arguments.auto_search = msg.get("value")
                print(f"[INFO] Importer {self.side}: auto_search set to {msg.get('value')}")

            elif msg.get("param") == "minThrRad":
                config.graphical_user_interface.pupil_processor.min_radius = msg.get("value")
                print(f"[INFO] Importer {self.side}: minR set to {msg.get('value')}")

            elif msg.get("param") == "maxThrRad":
                config.graphical_user_interface.pupil_processor.max_radius = msg.get("value")
                print(f"[INFO] Importer {self.side}: maxR set to {msg.get('value')}")

            elif msg.get("param") == "search_step":
                config.arguments.search_step = msg.get("value")
                print(f"[INFO] Importer {self.side}: search step set to {msg.get('value')}")
            elif msg.get("param") == "preview":
                config.preview = msg.get("value")
                print(f"[INFO] Importer {self.side}: Preview set to {msg.get('value')}")
            else:
                print(f"[INFO] Importer {self.side}: "
                    f"Unknown configuration parameter: {msg.get('param')}"
                )
        except (TypeError, ValueError, KeyError) as e:
            print(f"[WARN] Importer {self.side}: Failed to apply config: {e}")


    def _connect_shm(self, msg):
        """Setup the shared memory segment."""

        self.frame_shape = tuple(msg["frame_shape"])
        self.frame_dtype = np.dtype(msg["frame_dtype"])
        self.shm = SharedMemory(name=self.shared_memory_name)
        self.tracker_shm_is_closed_signal.clear()
        print(f"[INFO] Importer {self.side}: Shared memory initialized.")


    def _close_shm(self):
        """Close the shared memory segment."""
        try:
            self.shm.close()
            self.tracker_shm_is_closed_signal.set()
            print(f"[INFO] Importer {self.side}: Shared memory closed.")
        except (FileNotFoundError, PermissionError, OSError, BufferError) as e:
            print(f"[ERROR] Importer {self.side}: Error closing shared memory: {e}")


    def release(self) -> None:
        """Release the importer and shared memory."""

        print(f"[INFO] Importer {self.side}: cv.Importer.release() called")
        self._close_shm()
        #cv2.destroyAllWindows()
