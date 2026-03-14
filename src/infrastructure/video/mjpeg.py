"""
MJPEG streaming — serves the raw video feed to the browser as a multipart stream.
The browser displays this in an <img> tag. Zero JS needed for the video itself.
"""

import cv2
import time
import asyncio
from src.infrastructure.video.source import FrameBuffer

BOUNDARY = b"--frameboundary"


async def mjpeg_stream(buffer: FrameBuffer, target_fps: float = 25):
    """
    Async generator that yields MJPEG frames from the shared buffer.
    Used by FastAPI's StreamingResponse.
    """
    interval = 1.0 / target_fps

    while True:
        data = buffer.current()
        if data is not None:
            frame, _ = data
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (
                BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg.tobytes() + b"\r\n"
            )

        await asyncio.sleep(interval)
