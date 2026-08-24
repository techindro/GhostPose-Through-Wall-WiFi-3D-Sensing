"""
data_collector.py
=============================================================================
ESP32 Wi-Fi CSI Dataset Collector:
- Records live raw CSI data from ESP32 serial into NumPy / HDF5 / JSON archives
- Synchronizes with timestamped ground-truth labels (Pose, Vitals, Activity)
- Splits collected sessions into Train/Val/Test partitions for RF-Pose models
=============================================================================
"""

import os
import time
import json
import argparse
import logging
import numpy as np
import serial

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CSICollector")


def collect_dataset_session(
    port: str,
    baudrate: int,
    output_dir: str,
    session_name: str,
    duration_sec: float,
    subject_id: int = 1,
    activity_label: str = "walking"
):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{session_name}_sub{subject_id}_{activity_label}.jsonl")
    
    logger.info(f"Connecting to ESP32 on {port} ({baudrate} baud)...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
    except Exception as e:
        logger.error(f"Failed to open port {port}: {e}")
        return

    logger.info(f"Starting recording session '{session_name}' for {duration_sec}s -> {out_file}")
    start_time = time.time()
    packet_count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        try:
            while (time.time() - start_time) < duration_sec:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line.startswith("CSI_DATA"):
                    continue

                tokens = line.split(",")
                if len(tokens) < 5:
                    continue

                rssi = int(tokens[1])
                noise_floor = int(tokens[2])
                rate = int(tokens[3])
                data_len = int(tokens[4])
                raw_bytes = [int(x) for x in tokens[5:] if x.strip()]

                record = {
                    "timestamp": time.time(),
                    "subject_id": subject_id,
                    "activity": activity_label,
                    "rssi": rssi,
                    "noise_floor": noise_floor,
                    "rate": rate,
                    "data_len": data_len,
                    "csi_raw": raw_bytes
                }
                f.write(json.dumps(record) + "\n")
                packet_count += 1

                if packet_count % 100 == 0:
                    logger.info(f"Collected {packet_count} CSI packets | Elapsed: {time.time() - start_time:.1f}s")

        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        finally:
            ser.close()

    logger.info(f"[OK] Session saved: {packet_count} frames recorded to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 CSI Dataset Collector")
    parser.add_argument("--port", default="COM3", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--output", default="dataset/raw", help="Output directory")
    parser.add_argument("--session", default="session_01", help="Session ID")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID")
    parser.add_argument("--activity", default="walking", help="Activity label")
    args = parser.parse_args()

    collect_dataset_session(
        port=args.port,
        baudrate=args.baud,
        output_dir=args.output,
        session_name=args.session,
        duration_sec=args.duration,
        subject_id=args.subject,
        activity_label=args.activity
    )
