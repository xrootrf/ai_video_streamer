import cv2
import time
import asyncio
import queue
import requests
import numpy as np
from threading import Thread

from ultralytics import YOLO

from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import get_capabilities
from av import VideoFrame


# =========================================================
# CONFIG
# =========================================================

# MediaMTX
# MEDIAMTX_API = "http://127.0.0.1:9997/v3/paths/list"
# RTSP_BASE   = "rtsp://127.0.0.1:8554"
# WHIP_BASE   = "http://127.0.0.1:8889"

MEDIAMTX_API = "https://droneuse.com:9997/v3/paths/list"
RTSP_BASE   = "rtsp://0.0.0.0:8554"
WHIP_BASE   = "https://droneuse.com:8889"

MODEL_PATH = "OCT_1_PER_VE_OBB.pt"
CONF_THRESHOLD = 0.4
STREAM_SCAN_INTERVAL = 2.0


# =========================================================
# CLASSES & COLORS
# =========================================================

# MUST match training order
CUSTOM_CLASSES = ["fire", "other", "person", "smoke", "vehicle"]
# CUSTOM_CLASSES = ["person", "vehicle"]

# Disable class completely
DISABLED_CLASSES = {"other"}

# Bounding box colors (BGR)
CLASS_COLORS = {
    "fire":    (0, 0, 255),     # red
    "person":  (0, 255, 0),     # green
    "smoke":   (255, 255, 0),   # cyan
    "vehicle": (255, 0, 0),     # blue
}

# Text colors
TEXT_COLORS = {
    "fire":    (255, 255, 255),
    "person":  (0, 0, 0),
    "smoke":   (0, 0, 0),
    "vehicle": (255, 255, 255),
}


# =========================================================
# MEDIAMTX DISCOVERY
# =========================================================

def get_active_paths():
    try:
        r = requests.get(MEDIAMTX_API, timeout=2)
        items = r.json().get("items", [])

        paths = []
        for it in items:
            if it.get("ready") and "H264" in it.get("tracks", []):
                if "_ai" in it["name"] or  it["ready"] == False:
                    continue
                paths.append(it["name"])

        return paths

    except Exception as e:
        print("[MediaMTX]", e)
        return []


# =========================================================
# PER-STREAM PIPELINE
# =========================================================

class StreamPipeline:
    def __init__(self, name):
        self.name = name
        self.rtsp_url = f"{RTSP_BASE}/{name}"
        self.whip_url = f"{WHIP_BASE}/{name}_ai/whip"

        self.frame_q = queue.Queue(maxsize=4)
        self.result_q = queue.Queue(maxsize=4)

        self.model = YOLO(MODEL_PATH, task="obb")
        self.running = True

        print(f"[Pipeline] Starting for stream: {name}")

        Thread(target=self.capture_loop, daemon=True).start()
        Thread(target=self.infer_loop, daemon=True).start()
        asyncio.create_task(self.start_whip())


    # ---------------- CAPTURE ----------------

    def capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            try:
                self.frame_q.put(frame, block=False)
            except queue.Full:
                pass

        cap.release()


    # ---------------- INFERENCE + DRAW ----------------

    def infer_loop(self):
        while self.running:
            try:
                frame = self.frame_q.get(timeout=1)
            except queue.Empty:
                continue

            annotated = frame.copy()

            try:
                results = self.model(
                    frame,
                    conf=CONF_THRESHOLD,
                    device=0,
                    half=True,
                    verbose=False,
                )

                if results:
                    for r in results:
                        if r.obb is None:
                            continue

                        for i in range(len(r.obb.cls)):
                            cls_id = int(r.obb.cls[i])
                            conf = float(r.obb.conf[i])

                            if cls_id >= len(CUSTOM_CLASSES):
                                continue

                            cls_name = CUSTOM_CLASSES[cls_id]

                            # ---- DISABLE CLASS ----
                            if cls_name in DISABLED_CLASSES:
                                continue
                            # -----------------------

                            pts = (
                                r.obb.xyxyxyxy[i]
                                .reshape(4, 2)
                                .cpu()
                                .numpy()
                                .astype(int)
                            )

                            box_color  = CLASS_COLORS.get(cls_name, (255,255,255))
                            text_color = TEXT_COLORS.get(cls_name, (0,0,0))

                            # Draw OBB
                            cv2.polylines(
                                annotated,
                                [pts],
                                True,
                                box_color,
                                2
                            )

                            label = f"{cls_name} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                2
                            )

                            x, y = pts[0]

                            # Label background
                            cv2.rectangle(
                                annotated,
                                (x, y - th - 6),
                                (x + tw + 4, y),
                                box_color,
                                -1
                            )

                            # Label text
                            cv2.putText(
                                annotated,
                                label,
                                (x + 2, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                text_color,
                                2
                            )

                try:
                    self.result_q.put(annotated, block=False)
                except queue.Full:
                    pass

            except Exception as e:
                print(f"[Infer:{self.name}]", e)


    # ---------------- WHIP VIDEO TRACK ----------------

    class DetectionTrack(VideoStreamTrack):
        def __init__(self, q):
            super().__init__()
            self.q = q
            self.last = None

        async def recv(self):
            pts, tb = await self.next_timestamp()

            try:
                frame = self.q.get_nowait()
                self.last = frame
            except:
                frame = self.last if self.last is not None else np.zeros(
                    (480, 640, 3), dtype=np.uint8
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vf = VideoFrame.from_ndarray(rgb, format="rgb24")
            vf.pts = pts
            vf.time_base = tb
            return vf


    # ---------------- WHIP STREAM ----------------

    async def start_whip(self):
        while self.running:
            try:
                pc = RTCPeerConnection()
                track = self.DetectionTrack(self.result_q)

                transceiver = pc.addTransceiver(
                    "video",
                    direction="sendonly"
                )

                codecs = get_capabilities("video").codecs
                h264 = [c for c in codecs if c.mimeType == "video/H264"]
                transceiver.setCodecPreferences(h264)

                # CRITICAL: attach track without adding new m-line
                transceiver.sender.replaceTrack(track)

                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)

                async with ClientSession() as session:
                    async with session.post(
                        self.whip_url,
                        headers={"Content-Type": "application/sdp"},
                        data=pc.localDescription.sdp,
                    ) as resp:
                        answer = await resp.text()

                await pc.setRemoteDescription(
                    RTCSessionDescription(answer, "answer")
                )

                print(f"[WHIP] Streaming started for {self.name}")

                while self.running:
                    await asyncio.sleep(1)

            except Exception as e:
                print(f"[WHIP:{self.name}] Error:", e)
                print("[WHIP] Retrying in 5 seconds...")
                await asyncio.sleep(5)


# =========================================================
# STREAM MANAGER
# =========================================================

async def main():
    pipelines = {}

    print("[System] Multi-stream parallel inference started")

    while True:
        active = set(get_active_paths())

        for name in active - pipelines.keys():
            pipelines[name] = StreamPipeline(name)

        for name in list(pipelines.keys()):
            if name not in active:
                pipelines[name].running = False
                del pipelines[name]

        await asyncio.sleep(STREAM_SCAN_INTERVAL)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())

