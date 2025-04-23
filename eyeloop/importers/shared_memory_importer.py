import queue

# ---- TCP Configuration --------------------------------------------
class TCPConfig:
    host = '0.0.0.0' # Listen on all interfaces
    port = 65432     # Port for the TCP server

    static_ip_prefix = "192.168.1." # Static IP prefix for the device
    recv_buffer_size = 1024         # Buffer size for receiving data
    send_loop_interval = 0.001      # Interval for sending messages (in seconds)

    google_dns = "8.8.8.8"  # Used to check if the device is connected to the internet
    http_port = 80          # Port for HTTP requests (if needed)

    # Message priorities for the TCP server
    message_priorities = {
        'high': queue.Queue(), # Reserved for gyroscope data
        'medium': queue.Queue(), # Reserved for eye tracker data
        'low': queue.Queue() # Reserved for other messages
    }

tcp_config = TCPConfig()


# ---- Gyroscope Configuration --------------------------------------
class GyroscopeConfig:
    bus_num = 1         # I2C bus number (1 for Raspberry Pi 3 and later)
    addr = 0x6b         # I2C address of the gyroscope (L3GD20H)
    update_rate = 0.01  # in seconds (100 Hz)
    retry_attempts = 10  # Number of attempts to read data from the gyroscope

    # Register map
    reg_ctrl1 = 0x20
    ctrl1_enable = 0x0F

    # Registers for X, Y, Z axes
    reg_out_x_l = 0x28
    reg_out_y_l = 0x2A
    reg_out_z_l = 0x2C

gyroscope_config = GyroscopeConfig()


# ---- ESP32 Configuration ------------------------------------------
class ESP32Config:
    port="/dev/serial0" # Serial port for ESP32 (e.g., /dev/serial0 on Raspberry Pi)
    baudrate=115200 # Baud rate for the serial connection 
    timeout=1 # Timeout for the serial connection (in seconds)

    handshake_attempts = 3 # Number of attempts to perform the handshake
    handshake_interval_inner = 1 # Interval between handshake attempts (in seconds)
    handshake_interval_outer = 5 # Interval between outer handshake attempts (in seconds)
    handshake_message = "STATUS" # Handshake message to send to ESP32
    handshake_response = "ONLINE" # Expected response from ESP32 after handshake

    send_attempts = 3 # Number of attempts to send the focal distance

esp32_config = ESP32Config()


# ---- Tracker Configuration ------------------------------------
class TrackerConfig:
    frame_provider_max_fps = 15 # Maximum FPS for the frame provider
    jpeg_quality = 75  # JPEG encoding quality (0-100)
    sync_timeout = 1.0  # Timeout for EyeLoop response in seconds
    #index = 0  # Only used for fallback/testing
    preview_fps = 5  # FPS for preview stream
    
    crop_left = ((0.0, 0.5), (0.0, 1.0))  # Relative region (x1, x2, y1, y2) for the left eye
    crop_right = ((0.5, 1.0), (0.0, 1.0))  # Relative region (x1, x2, y1, y2) for the right eye

    process_launch_time = 0.4  # Time to wait for the EyeLoop process to stabilize (in seconds)

    sharedmem_name_left = "eye_left_frame"  # Shared memory buffer name for left eye
    sharedmem_name_right = "eye_right_frame"  # Shared memory buffer name for right eye
    
    importer_name = "shared_memory_importer"  # Importer name for the EyeLoop process
    
    sync_timeout = 0.2  # Timeout for EyeLoop response in seconds
    queue_timeout = 0.005  # Timeout for queue operations in seconds

    eyeloop_health_check_interval = 3  # Interval for health check of the eyeloop processes in seconds

    use_test_video = False  # Use saved video instead of live camera
    test_video_path = "test_video/test_video.mp4"  # Path to test video
    test_video_resolution = (1920, 1080)  # Hardcoded resolution, must be changed in the code if needed
    test_video_channels = 3  # Number of channels in the test video, must be changed in the code if needed
    

tracker_config = TrackerConfig()


# ---- Camera Configuration ----------------------------------------
class CameraManagerConfig:

    width = 200
    height = 100

    focus = 2.5  # Only used if autofocus is False
    exposure_time = 10000  # In microseconds
    analogue_gain = 2.0  # Brightness boost
    af_mode = 0  # 0 = manual, 1 = auto

    capture_retries = 3  # Number of attempts to capture a frame

camera_manager_config = CameraManagerConfig()

# ---- Health Monitor Configuration --------------------------------
class HealthMonitorConfig:

    check_interval = 2  # Interval for health check in seconds
    monitored_components = []
