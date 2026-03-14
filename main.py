"""
ParkEdge Vision — main app.

Start the server:
    python main.py
    python main.py --port 8080

Then in a separate terminal, start the feed simulator:
    python -m tools.simulate_feed videos/sample_short.mp4 --loop
"""

import argparse
import logging
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for name in ("httpx", "httpcore", "paddle", "paddleocr"):
    logging.getLogger(name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="ParkEdge Vision")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("src.presentation.server:app", host=args.host, port=args.port, reload=False, access_log=False)


if __name__ == "__main__":
    main()
