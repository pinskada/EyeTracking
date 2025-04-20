import numpy as np
import cv2

def encode_binary_float64_as_png(binary_image: np.ndarray) -> bytes:
    """
    Convert a float64 binary image (0.0 or 1.0) to PNG bytes.
    """
    if binary_image.dtype != np.float64:
        raise ValueError("Expected float64 image")

    # Step 1: Scale 0.0/1.0 to 0/255
    scaled = (binary_image * 255).astype(np.uint8)

    # Step 2: Encode as PNG
    success, encoded = cv2.imencode(".png", scaled)
    if not success:
        raise RuntimeError("PNG encoding failed")

    return encoded.tobytes()
