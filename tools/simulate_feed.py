"""
Debug tool — standalone process that replays a video file as a live feed.
Sends JPEG frames over HTTP to the main app's /ingest endpoint.
Runs at the video's native FPS regardless of app processing speed.

Usage:
    python -m tools.simulate_feed videos/sample_short.mp4
    python -m tools.simulate_feed videos/sample_short.mp4 --fps 15 --loop
    python -m tools.simulate_feed videos/sample_short.mp4 --url http://localhost:8000
"""

import argparse
import logging
import sys
import time

import cv2
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [simulate] %(message)s",
)
logger = logging.getLogger(__name__)


def simulate(video_path: str, ingest_url: str, fps_override: float | None = None, loop: bool = False):
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open: %s", video_path)
            sys.exit(1)

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fps = fps_override or native_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = 1.0 / fps

        logger.info("Playing %s at %.1f FPS (%d frames) → %s", video_path, fps, total, ingest_url)

        frame_index = 0
        errors = 0
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break

            # Encode as JPEG and POST to the app — fire and forget
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                requests.post(
                    ingest_url,
                    data=jpeg.tobytes(),
                    headers={"Content-Type": "image/jpeg", "X-Frame-Index": str(frame_index)},
                    timeout=0.5,
                )
                errors = 0
            except Exception:
                errors += 1
                if errors == 1:
                    logger.warning("App not reachable at %s — frames will be dropped", ingest_url)
                elif errors % 100 == 0:
                    logger.warning("Still can't reach app (%d consecutive failures)", errors)

            frame_index += 1

            # Sleep to maintain real-time playback — never wait for the app
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        duration = frame_index / fps if fps else 0
        logger.info("Finished. %d frames in %.1fs", frame_index, duration)

        if not loop:
            break
        logger.info("Looping...")


def main():
    parser = argparse.ArgumentParser(description="Simulate a live drone feed → sends to ParkEdge Vision app")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS (default: video native)")
    parser.add_argument("--loop", action="store_true", help="Loop the video forever")
    parser.add_argument("--url", default="http://localhost:8000", help="App base URL")
    args = parser.parse_args()

    ingest_url = args.url.rstrip("/") + "/ingest"
    simulate(args.video, ingest_url, fps_override=args.fps, loop=args.loop)


if __name__ == "__main__":
    main()
