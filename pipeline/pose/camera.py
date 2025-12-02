# pipeline/pose/camera.py
import cv2
import time
import logging

log = logging.getLogger("camera")

class Camera:
    def __init__(self, index=0, resolution=(1280, 720), fps=15):
        self.index = index
        self.cap = cv2.VideoCapture(index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        time.sleep(0.2)  # allow warm-up

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} could not be opened")
        log.info(f"Camera {index} opened with resolution {resolution} at {fps} FPS")
        self.last=time.time()
        self.fps=fps

    def read(self):
        """Reads a valid frame, retries if needed."""
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                return frame

            time.sleep(0.05)

        raise RuntimeError("Camera read failed after retries")
    

    def release(self):
        """Safely release camera."""
        try:
            if self.cap.isOpened():
                self.cap.release()
        except:
            pass
