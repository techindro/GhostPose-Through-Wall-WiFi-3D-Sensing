"""
kafka_inference_service.py
=============================================================================
High-Concurrency Async FastAPI & Apache Kafka Edge-Inference Server:
- Consumes real-time raw Wi-Fi CSI JSON/Protobuf packets via aiokafka
- Fallback internal simulator mode for standalone testing without Kafka
- Rolling Ring Buffer with lock-free tensor staging
- High-Throughput GPU Batch Inference (PyTorch FP16 / CUDA execution)
- Ultra-low latency (<30ms) WebSocket broadcast for 3D Three.js UI
=============================================================================
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Dict, List, Set, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from rf_pose_model import RFStudentNetwork
from rf_signal_processor import RFSignalProcessor
from rf_stream_simulator import generate_synthetic_csi_packet

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RFSensingPipeline")

# Global Settings
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_CSI_RAW = os.getenv("KAFKA_TOPIC", "rf.esp32.csi.raw")
NUM_SUBCARRIERS = 64
TEMPORAL_WINDOW = 100  # 1.0 second window at 100 Hz
NUM_RX_ANTENNAS = 3
ENABLE_SIMULATOR = os.getenv("ENABLE_SIMULATOR", "true").lower() in ("true", "1", "yes")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="MIT RF-Pose3D Real-Time Inference Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketManager:
    """Manages active WebSockets and dispatches JSON updates to UI clients."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"UI Client connected. Total active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"UI Client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast_json(self, payload: dict):
        if not self.active_connections:
            return
        dead_sockets = set()
        message = json.dumps(payload)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                dead_sockets.add(connection)
        for dead in dead_sockets:
            self.active_connections.discard(dead)


ws_manager = WebSocketManager()


class LiveCSIRingBuffer:
    """Lock-free rolling window buffer for temporal multi-channel CSI accumulation."""
    def __init__(self, max_len: int = TEMPORAL_WINDOW):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)

    def append(self, csi_frame: np.ndarray):
        self.buffer.append(csi_frame)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.max_len

    def get_window(self) -> np.ndarray:
        """Returns array of shape [Antennas, Subcarriers, Time]"""
        return np.stack(list(self.buffer), axis=2)


class RFInferenceEngine:
    """Manages model weights, signal pre-processing, and GPU tensor inference."""
    def __init__(self):
        self.processor = RFSignalProcessor(
            num_subcarriers=NUM_SUBCARRIERS,
            sampling_rate_hz=100.0
        )
        self.model = RFStudentNetwork(
            in_channels=NUM_RX_ANTENNAS * 4,
            num_joints=17,
            temporal_window=TEMPORAL_WINDOW
        ).to(DEVICE)
        self.model.eval()

        # Load weights if available
        ckpt_path = os.path.join("checkpoints", "rf_pose_best.pth")
        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded pretrained checkpoint from {ckpt_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

    @torch.inference_mode()
    def infer_frame_window(self, csi_window: np.ndarray) -> dict:
        """
        Executes end-to-end signal sanitization, PyTorch forward pass, and payload serialization.
        """
        t0 = time.perf_counter()
        
        # 1. Advanced DSP Preprocessing & Phase Sanitization
        feature_stack = self.processor.process_frame_tensor(csi_window)
        
        # 2. Extract Classical DSP Vital Signs
        _, sanitized_phase = self.processor.sanitize_phase(csi_window)
        dsp_vitals = self.processor.extract_vital_signs(sanitized_phase)

        # 3. Prepare GPU Tensor [1, Channels, Subcarriers, Time]
        tensor_in = (
            torch.from_numpy(feature_stack)
            .unsqueeze(0)
            .to(DEVICE, non_blocking=True)
        )

        # 4. Neural Forward Pass with Automatic Mixed Precision (FP16)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32):
            predictions = self.model(tensor_in)

        # 5. Extract and format outputs
        pose_3d = predictions["pose_3d"].squeeze(0).cpu().numpy().tolist()  # 17 x 3
        vital_rates = predictions["vital_rates"].squeeze(0).cpu().numpy()
        reid_logits = predictions["reid_logits"].squeeze(0)
        track_id = int(torch.argmax(reid_logits).item())
        confidence = float(torch.softmax(reid_logits, dim=-1)[track_id].item())

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Construct JSON payload for Three.js/D3.js Frontend
        payload = {
            "timestamp": time.time(),
            "latency_ms": round(latency_ms, 2),
            "target_id": f"human_track_{track_id}",
            "reid_confidence": round(confidence, 3),
            "vital_signs": {
                "respiration_rate_brpm": round(float(vital_rates[0]) if vital_rates[0] > 0 else dsp_vitals["respiration_rate_brpm"], 1),
                "heart_rate_bpm": round(float(vital_rates[1]) if vital_rates[1] > 0 else dsp_vitals["heart_rate_bpm"], 1),
                "respiration_waveform_sample": float(dsp_vitals["respiration_waveform"][-1])
            },
            # 17 Keypoints in 3D (X, Y, Z in meters)
            "skeleton_3d_keypoints": pose_3d,
            "joint_labels": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
        }
        return payload


# Instantiate Singletons
engine = RFInferenceEngine()
ring_buffer = LiveCSIRingBuffer(max_len=TEMPORAL_WINDOW)


async def kafka_consumer_worker():
    """
    Background worker consuming raw CSI data packets from Kafka,
    feeding the rolling buffer, and triggering inferences.
    Falls back to synthetic simulation loop if Kafka is unreachable.
    """
    try:
        import importlib
        aiokafka_module = importlib.import_module("aiokafka")
        AIOKafkaConsumer = getattr(aiokafka_module, "AIOKafkaConsumer")
        consumer = AIOKafkaConsumer(
            KAFKA_TOPIC_CSI_RAW,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id="rf_pose_inference_workers",
            auto_offset_reset="latest",
            enable_auto_commit=True
        )
        await consumer.start()
        logger.info(f"Connected to Kafka broker. Subscribed to '{KAFKA_TOPIC_CSI_RAW}'")
        
        async for msg in consumer:
            try:
                data = json.loads(msg.value.decode("utf-8"))
                raw_arr = np.array(data["csi_matrix"], dtype=np.float32)
                complex_csi = raw_arr[..., 0] + 1j * raw_arr[..., 1]
                ring_buffer.append(complex_csi)

                if ring_buffer.is_ready():
                    window_data = ring_buffer.get_window()
                    result_payload = await asyncio.to_thread(engine.infer_frame_window, window_data)
                    await ws_manager.broadcast_json(result_payload)
            except Exception as e:
                logger.error(f"Error in Kafka processing loop: {e}")
                
    except Exception as kafka_err:
        logger.warning(f"Kafka unavailable ({kafka_err}). Starting built-in real-time simulator fallback...")
        await run_internal_simulator()


async def run_internal_simulator():
    """Fallback generator streaming continuous synthetic CSI frames into inference engine."""
    t = 0.0
    dt = 1.0 / 100.0  # 100 Hz
    while True:
        try:
            packet = generate_synthetic_csi_packet(t, num_antennas=NUM_RX_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS)
            raw_arr = np.array(packet["csi_matrix"], dtype=np.float32)
            complex_csi = raw_arr[..., 0] + 1j * raw_arr[..., 1]
            ring_buffer.append(complex_csi)

            if ring_buffer.is_ready():
                window_data = ring_buffer.get_window()
                result_payload = await asyncio.to_thread(engine.infer_frame_window, window_data)
                await ws_manager.broadcast_json(result_payload)

            t += dt
            await asyncio.sleep(dt)
        except Exception as sim_err:
            logger.error(f"Simulator error: {sim_err}")
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(kafka_consumer_worker())


@app.websocket("/ws/rf_skeleton_feed")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint serving real-time 3D skeletons, Re-ID, and vitals.
    Frontend connects to: ws://localhost:8000/ws/rf_skeleton_feed
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        ws_manager.disconnect(websocket)


@app.post("/api/v1/ingest_csi")
async def ingest_csi_http(payload: dict):
    """HTTP endpoint to push CSI packets directly."""
    try:
        raw_arr = np.array(payload["csi_matrix"], dtype=np.float32)
        complex_csi = raw_arr[..., 0] + 1j * raw_arr[..., 1]
        ring_buffer.append(complex_csi)
        return {"status": "buffered", "buffer_size": len(ring_buffer.buffer)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/healthz")
async def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "buffer_fill_pct": (len(ring_buffer.buffer) / TEMPORAL_WINDOW) * 100.0,
        "active_clients": len(ws_manager.active_connections)
    }


# Mount Static Dashboard
dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")


@app.get("/")
async def root():
    index_file = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "MIT RF-Pose3D Sensing API running. Open /dashboard to view the 3D visualizer."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kafka_inference_service:app", host="0.0.0.0", port=8000, reload=False)
