import cv2
import asyncio
import queue
import time
from threading import Thread
from aiohttp import ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import get_capabilities
from av import VideoFrame
from ultralytics import YOLO
import numpy as np


# ---------------------------------------------------------
#  CONFIG (EDIT THESE)
# ---------------------------------------------------------
print("running")
# WHIP_ENDPOINT_URL = "https://droneuse.com:8889/6620e41e2ca5b155a6dea465_ai/whip"  # <--- CHANGE THIS
WHIP_ENDPOINT_URL = "http://0.0.0.0:8889/amar_ai/whip"  # <--- CHANGE THIS

RTSP_PIPELINE = (
    "rtspsrc location=rtsp://0.0.0.0:8554/stream latency=0 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink"
)

MODEL_PATH =YOLO("OCT_1_PER_VE_OBB.pt",task="obb")   # <--- CHANGE THIS

# CUSTOM CLASSES — NOT COCO!
CUSTOM_CLASSES = ["person", "vehicle"]   # <--- PUT YOUR CLASSES HERE

CONF_THRESHOLD = 0.40

NUM_WORKERS = 2   # number of PyTorch inference threads

# ---------------------------------------------------------
#  THREAD QUEUES
# ---------------------------------------------------------

frame_queue = queue.Queue(maxsize=4)
result_queue = queue.Queue(maxsize=4)

# ---------------------------------------------------------
#  DETECTION WORKER (PyTorch)
# ---------------------------------------------------------

class TorchOBBWorker(Thread):
    def __init__(self, worker_id, model_path, custom_classes):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.model = YOLO(model_path, task="obb")
        self.custom_classes = custom_classes
        print(f"[Worker {worker_id}] OBB model loaded.")

    def run(self):
        print(f"[Worker {self.worker_id}] Starting OBB detection loop...")
        while True:
            try:
                frame, ts = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                results = self.model(frame, conf=CONF_THRESHOLD, verbose=True)
                if results is None or len(results) == 0:
                    continue

                annotated = frame.copy()

                for r in results:
                    obb = r.obb
                    if obb is None:
                        continue

                    xyxyxyxy_list = obb.xyxyxyxy   # [N, 8]
                    cls_list = obb.cls
                    conf_list = obb.conf

                    for i in range(len(cls_list)):
                        cls = int(cls_list[i])
                        conf = float(conf_list[i])

                        # -------- FIX IS HERE --------
                        pts = (
                            xyxyxyxy_list[i]
                            .reshape(4, 2)
                            .cpu()
                            .numpy()
                            .astype(int)
                        )
                        # --------------------------------

                        cls_name = (
                            self.custom_classes[cls]
                            if cls < len(self.custom_classes)
                            else f"class_{cls}"
                        )

                        # Draw box
                        cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)

                        # Draw label
                        x_text, y_text = pts[0]
                        cv2.putText(
                            annotated,
                            f"{cls_name} {conf:.2f}",
                            (x_text, y_text - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                try:
                    result_queue.put((annotated, ts), block=False)
                except queue.Full:
                    pass

            except Exception as e:
                print("[Worker Error]", e)
                continue



# ---------------------------------------------------------
#  CAPTURE THREAD
# ---------------------------------------------------------

def capture_loop():
    # cap = cv2.VideoCapture(RTSP_PIPELINE, cv2.CAP_GSTREAMER)
    
    # cap = cv2.VideoCapture("rtsp://0.0.0.0:8554/6620e41e2ca5b155a6dea465")
    cap = cv2.VideoCapture("rtsp://0.0.0.0:8554/amar")
    if not cap.isOpened():
        print("ERROR: Cannot open RTSP stream.")
        return

    print("[Capture] Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        ts = time.time()

        try:
            frame_queue.put((frame.copy(), ts), block=False)
        except queue.Full:
            pass


# ---------------------------------------------------------
#  WHIP VIDEO STREAM TRACK
# ---------------------------------------------------------

class DetectionVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.last_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        try:
            frame, ts = result_queue.get(timeout=0.02)
            self.last_frame = frame
        except queue.Empty:
            if self.last_frame is None:
                black = np.zeros((480,640,3), dtype=np.uint8)
                frame = black
            else:
                frame = self.last_frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        avf = VideoFrame.from_ndarray(rgb, format="rgb24")
        avf.pts = pts
        avf.time_base = time_base
        return avf


# ---------------------------------------------------------
#  WHIP STREAMER
# ---------------------------------------------------------

async def whip_stream():
    pc = RTCPeerConnection()
    track = DetectionVideoTrack()

    codecs = get_capabilities("video").codecs
    h264 = [c for c in codecs if c.mimeType == "video/H264"]

    tx = pc.addTransceiver(track, direction="sendonly")
    tx.setCodecPreferences(h264)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with ClientSession() as s:
        async with s.post(
            WHIP_ENDPOINT_URL,
            headers={"Content-Type": "application/sdp"},
            data=offer.sdp
        ) as resp:
            answer_sdp = await resp.text()
            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await pc.setRemoteDescription(answer)
            print("[WHIP] Streaming started.")

            while True:
                await asyncio.sleep(0.5)


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

async def main():
    # Start capture
    cap_thread = Thread(target=capture_loop, daemon=True)
    cap_thread.start()

    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        w = TorchOBBWorker(i, MODEL_PATH, CUSTOM_CLASSES)
        w.start()
        workers.append(w)

    # Start WHIP
    await whip_stream()


if __name__ == "__main__":
    asyncio.run(main())
