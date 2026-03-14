"""
Presentation layer — FastAPI server.
- GET  /feed         → MJPEG stream (raw video, displayed in <img>)
- WS   /ws           → overlay data (bboxes, detections) as JSON
- GET  /detections   → current confirmed + pending
- POST /reset        → clear tracker
- GET  /             → static HTML UI
"""

import asyncio
import json
import logging
import queue
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse

from src.domain.detection import FrameResult
from src.application.scanner import ScannerService
from src.application.tracker import DetectionTracker
from src.infrastructure.ocr.paddle_ocr import PaddleOCREngine
from src.infrastructure.video.source import FrameBuffer
from src.infrastructure.video.mjpeg import mjpeg_stream

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ── Shared state (wired up in lifespan) ──
frame_buffer = FrameBuffer()
tracker = DetectionTracker(min_hits=2, expiry_seconds=30.0)
scanner: ScannerService | None = None
ws_clients: set[WebSocket] = set()
msg_queue: queue.Queue[str] = queue.Queue()


def _on_scan_result(result: FrameResult, newly_confirmed: list):
    """Called by ScannerService from its OCR thread — puts JSON on the queue."""
    payload = json.dumps({
        "type": "frame",
        "frameIndex": result.frame_index,
        "visible": [
            {"value": d.value, "kind": d.kind, "confidence": round(d.confidence, 3)}
            for d in result.detections
        ],
        "newlyConfirmed": [v.to_dict() for v in newly_confirmed],
        "totalConfirmed": len(tracker.confirmed),
        "totalPending": len(tracker.pending),
    })
    msg_queue.put(payload)


async def _poll_and_broadcast():
    """Async task: drains the thread-safe queue → sends to all WS clients."""
    while True:
        try:
            while True:
                try:
                    message = msg_queue.get_nowait()
                except queue.Empty:
                    break
                dead = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead.add(ws)
                for d in dead:
                    ws_clients.discard(d)
        except Exception as e:
            logger.error("Broadcast error: %s", e)
        await asyncio.sleep(0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scanner
    logger.info("Starting OCR engine...")
    ocr_engine = PaddleOCREngine(ocr_max_width=1280)
    scanner = ScannerService(frame_buffer, ocr_engine, tracker, scan_interval_ms=1000)
    scanner.on_result(_on_scan_result)
    broadcast_task = asyncio.create_task(_poll_and_broadcast())
    logger.info("Server ready. Waiting for video feed...")
    yield
    scanner.stop()
    broadcast_task.cancel()


app = FastAPI(title="ParkEdge Vision", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Endpoints ──

@app.post("/ingest")
async def ingest_frame(request: Request):
    """
    Receive a JPEG frame from an external source (simulator, HDMI capture, etc).
    The source POSTs raw JPEG bytes here at its own pace — the app never blocks it.
    """
    body = await request.body()
    frame_index = int(request.headers.get("X-Frame-Index", frame_buffer.frame_index + 1))
    arr = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        frame_buffer.write(frame, frame_index)
    return {"ok": True}


@app.get("/feed")
async def video_feed():
    """Raw MJPEG video stream — browser shows this untouched in an <img> tag."""
    return StreamingResponse(
        mjpeg_stream(frame_buffer, target_fps=25),
        media_type="multipart/x-mixed-replace; boundary=--frameboundary",
    )


@app.get("/detections")
async def detections():
    return {
        "confirmed": [v.to_dict() for v in tracker.confirmed],
        "pending": [v.to_dict() for v in tracker.pending],
    }


@app.post("/reset")
async def reset():
    tracker.reset()
    msg_queue.put(json.dumps({"type": "reset"}))
    return {"ok": True}


@app.post("/scanner/{action}")
async def scanner_toggle(action: str):
    """Start or stop the OCR scanner. POST /scanner/start or /scanner/stop"""
    if scanner is None:
        return JSONResponse({"error": "Scanner not initialized"}, 503)
    if action == "start":
        scanner.start()
        return {"ok": True, "scanning": True}
    elif action == "stop":
        scanner.stop()
        return {"ok": True, "scanning": False}
    return JSONResponse({"error": "Use /scanner/start or /scanner/stop"}, 400)


@app.get("/status")
async def status():
    return {
        "feed_active": frame_buffer.has_frame,
        "scanning": scanner.running if scanner else False,
        "confirmed": len(tracker.confirmed),
        "pending": len(tracker.pending),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("WS client connected (%d total)", len(ws_clients))

    await ws.send_text(json.dumps({
        "type": "state",
        "confirmed": [v.to_dict() for v in tracker.confirmed],
        "pending": [v.to_dict() for v in tracker.pending],
        "feedActive": frame_buffer.has_frame,
        "scanning": scanner.running if scanner else False,
    }))

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "reset":
                tracker.reset()
                await ws.send_text(json.dumps({"type": "reset"}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)
        logger.info("WS client disconnected (%d total)", len(ws_clients))


# ── Static files (catch-all, must be last) ──
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
