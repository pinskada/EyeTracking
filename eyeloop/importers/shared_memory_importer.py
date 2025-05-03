from multiprocessing.shared_memory import SharedMemory
from queue import Empty
import numpy as np
import cv2
import time
import threading
import eyeloop.config as config
from eyeloop.importers.importer import IMPORTER

class Importer():
    def __init__(self) -> None:
        self.scale = config.arguments.scale
        
        # Wait for init command via command_queue
        self.shared_memory_name = config.arguments.sharedmem
        self.side = config.arguments.side
        self.frame_shape = None
        self.frame_dtype = None
        self.frame = None
        self.print_status = False
        self.current_frame_id = 0

        self.command_queue = config.command_queue
        self.sync_queue = config.sync_queue

        self._initialize_from_command_queue()

    def _initialize_from_command_queue(self):
        print(f"[INFO] Importer {self.side}: Waiting for init command...")
        while True:
            try:
                msg = self.command_queue.get(timeout=1)
                if msg.get("type") == "memory":
                    self._setup_shared_memory(msg)
                    break
            except Empty:
                
                continue

    def route(self):
        self.first_frame()
        time.sleep(0.3)  # Wait for the engine to stabilize
        print(f"[INFO] Importer {self.side}: Starting routing thread...\n")
        self.proceed_thread = threading.Thread(target=self.proceed())
        self.proceed_thread.run() # Start the frame provider
        
    def proceed(self) -> None:
        while True:
            self.load_command_queue()

            if self.shm is None:
                while self.shm is None:
                    self.load_command_queue()
                    time.sleep(0.001)

            if self.current_frame_id % 50 == 0 and self.print_status:
                print(f"[INFO] Importer {self.side}: Current frame ID: {self.current_frame_id}\n")
        
            try:
                msg = self.sync_queue.get(timeout=1)
                if msg.get("type") == 'frame_id':
                    self.print_status = True
                    self.current_frame_id = msg.get('frame_id')
                    frame = self.route_frame()
                    config.engine.iterate(frame)
            except Empty:
                self.print_status = False
                time.sleep(0.001)

    def first_frame(self) -> None:
        print(f"[INFO] Importer {self.side}: First frame processing started.\n")
        
        try:
            frame = self.route_frame()
        except Exception as e:
            print(f"[ERROR] Importer {self.side}: Error in first frame processing: {e}")
            return

        print("Arming the engine...")
        config.engine.arm(
            width=self.frame_shape[0],
            height=self.frame_shape[1],
            image=frame
        )   

    def route_frame(self) -> None:
        return np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shm.buf).copy()

    def load_command_queue(self):
        try:
            msg = self.command_queue.get_nowait()
            if msg.get("type") == "memory":
                self._setup_shared_memory(msg)
            elif msg.get("type") == "detach":
                self.close_memory()
            elif msg.get("type") == "close":
                print(f"[INFO] Importer {self.side}: Closing shared memory...")
                self.release()
            elif msg.get("type") == "config":
                self.configure(msg)
            elif msg.get("type") == "preview":
                config.preview = msg.get("value")
                print(f"[INFO] Importer {self.side}: Preview set to {msg.get('value')}")
            else:
                print(f"[INFO] Importer {self.side}: Unknown command: {msg.get('type')}")
        except Exception:
            pass        

    def configure(self, msg):
        """ Configure the run-time parameters based on the received message."""

        if  msg.get("param") == "threshold":
            config.graphical_user_interface.binary_threshold = msg.get("value")
            print(f"[INFO] Importer {self.side}: thresholsd set to {msg.get('value')}")

        elif msg.get("param") == "blur":
            config.graphical_user_interface.blur = msg.get("value")
            print(f"[INFO] Importer {self.side}: blur set to {msg.get('value')}")

        elif msg.get("param") == "auto_search":
            config.arguments.auto_search = msg.get("value")
            print(f"[INFO] Importer {self.side}: auto_search set to {msg.get('value')}")

        elif msg.get("param") == "minR":
            config.graphical_user_interface.min_radius_threshold.min_radius = msg.get("value")
            print(f"[INFO] Importer {self.side}: minR set to {msg.get('value')}")

        elif msg.get("param") == "maxR":
            config.graphical_user_interface.max_radius_threshold.max_radius = msg.get("value")
            print(f"[INFO] Importer {self.side}: maxR set to {msg.get('value')}")

        elif msg.get("param") == "step":
            config.arguments.search_step = msg.get("value")
            print(f"[INFO] Importer {self.side}: search step set to {msg.get('value')}")

        else:
            print(f"[INFO] Importer {self.side}: Unknown configuration parameter: {msg.get('param')}")
  
    def _setup_shared_memory(self, msg):

        self.frame_shape = tuple(msg["frame_shape"])
        self.frame_dtype = np.dtype(msg["frame_dtype"])
        self.shm = SharedMemory(name=self.shared_memory_name)
        print(f"[INFO] Importer {self.side}: Memory initialized with {self.shared_memory_name}, shape {self.frame_shape}, dtype {self.frame_dtype}\n")

    def close_memory(self):
        try:
            self.shm.close()
            self.shm = None
        except Exception as e:
            print(f"[ERROR] Importer {self.side}: Error closing shared memory: {e}")

    def release(self) -> None:
        print(f"[INFO] Importer {self.side}: cv.Importer.release() called")
        self.close_memory()
        #cv2.destroyAllWindows()
     