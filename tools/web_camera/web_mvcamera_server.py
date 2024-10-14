import time
import cv2
import uvicorn
import threading
import queue
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import mvsdk
from third_party.mvcam.vcamera import vCameraSystem
from loguru import logger

# setting the buffer size
BUFFER_SIZE = 200
frame_queue = queue.Queue(maxsize=BUFFER_SIZE)

EXPOSURE_MS = 0
FRAME_SPEED = 60
TRIGGER = 0
RESOLUTION_PRESET = 3
IDX = 0


def capture_frames():
    DevList = mvsdk.CameraEnumerateDevice()
    if len(DevList) < 1:
        logger.error("No MindVision camera was found!")
        return

    cam_sys = vCameraSystem(exposure_ms=EXPOSURE_MS,
                            trigger=TRIGGER,
                            resolution_preset=RESOLUTION_PRESET,
                            frame_speed=FRAME_SPEED)
    camera = cam_sys[IDX]
    with camera as c:
        while True:
            frame = None
            while frame is None:
                frame = c.read()
            # convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # encode image to JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            # transform JPEG format image to byte stream
            frame = jpeg.tobytes()
            if frame_queue.full():
                frame_queue.get()  # discard the oldest frame
            frame_queue.put(frame)
            time.sleep(1 / FRAME_SPEED)


def generate_video_stream():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.001)


app = FastAPI()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # start the thread to capture frames
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8123)