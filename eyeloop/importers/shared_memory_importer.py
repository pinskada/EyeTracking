from multiprocessing.shared_memory import SharedMemory
from queue import Empty
import numpy as np
import cv2
import time
import eyeloop.config as config
from eyeloop.importers.importer import IMPORTER

class Importer(IMPORTER):
    def __init__(self) -> None:
        self.scale = config.arguments.scale

        # Wait for init command via command_queue
        self.shared_memory_name = config.arguments.sharedmem
        self.side = config.arguments.side
        self.frame_shape = None
        self.frame_dtype = None
        self.frame = None
        self.current_frame_id = None

        self.command_queue = config.command_queue
        self.sync_queue = config.sync_queue

        self._initialize_from_command_queue()

    def _initialize_from_command_queue(self):
        print("[Importer] Waiting for init command...")
        while True:
            try:
                msg = self.command_queue.get(timeout=1)
                if msg.get("type") == "memory":
                    self._setup_shared_memory(msg)
                    break
            except Empty:
                
                continue

    def _setup_shared_memory(self, msg):
        self.frame_shape = tuple(msg["frame_shape"])
        self.frame_dtype = np.dtype(msg["frame_dtype"])
        self.shm = SharedMemory(name=self.shared_memory_name)
        print(f"[Importer] Memory initialized with {self.shared_memory_name}, shape {self.frame_shape}, dtype {self.frame_dtype}")

    def first_frame(self) -> None:
        print("[Importer] First frame processing started.")
        
        try:
            frame = self.route_frame()
        except Exception as e:
            print(f"[Importer] Error in first frame processing: {e}")
            return

        self.arm(
            width=self.frame_shape[0],
            height=self.frame_shape[1],
            image=frame
        )   

    def route(self):
        self.first_frame()
        while True:
            self.load_command_queue()
            try:
                msg = self.sync_queue.get(timeout=1)
                if msg.get("type") == "frame_id":
                    self.current_frame_id = msg.get("frame_id")
                    frame = self.route_frame()
                    self.proceed(frame)
            except Empty:
                continue

    def route_frame(self) -> None:
        return np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shm.buf).copy()

    def get_current_frame_id(self):
        return self.current_frame_id

    def load_command_queue(self):
        try:
            msg = self.command_queue.get_nowait()
            if msg.get("type") == "memory":
                self._setup_shared_memory(msg)
            elif msg.get("type") == "close":
                print("[Importer] Closing shared memory...")
                self.release()
            elif msg.get("type") == "config":
                self.configure(msg)
            elif msg.get("type") == "preview":
                config.preview = msg.get("value")
                print(f"[Importer] Preview set to {msg.get('value')}")
            else:
                print(f"[Importer] Unknown command: {msg.get('type')}")
        except self.command_queue.Empty:
            pass        

    def configure(self, msg):
        """ Configure the run-time parameters based on the received message."""

        if  msg.get("param") == "threshold":
            config.graphical_user_interface.binary_threshold = msg.get("value")
        elif msg.get("param") == "blur":
            config.graphical_user_interface.blur = msg.get("value")
        elif msg.get("param") == "auto_search":
            config.arguments.auto_search = msg.get("value")
            print(f"[Importer] auto_search set to {msg.get('value')}")
        elif msg.get("param") == "minR":
            config.graphical_user_interface.min_radius_threshold.min_radius = msg.get("value")
            print(f"[Importer] minR set to {msg.get('value')}")
        elif msg.get("param") == "maxR":
            config.graphical_user_interface.max_radius_threshold.max_radius = msg.get("value")
            print(f"[Importer] maxR set to {msg.get('value')}")
        elif msg.get("param") == "step":
            config.arguments.search_step = msg.get("value")
            print(f"[Importer] search step set to {msg.get('value')}")
        else:
            print(f"[Importer] Unknown configuration parameter: {msg.get('param')}")
  

    def release(self) -> None:
        print(f"cv.Importer.release() called")
        if self.capture is not None:
            self.capture.release()

        self.route_frame = None
        cv2.destroyAllWindows()
        super().release()      