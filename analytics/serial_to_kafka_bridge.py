"""
serial_to_kafka_bridge.py
=============================================================================
ESP32 Serial-to-Kafka Real-Time Gateway:
- Ingests raw CSI Serial stream from ESP32 receiver.ino
- Parses line protocol: "CSI_DATA,rssi,noise_floor,rate,len,b0,b1,b2,..."
- Unpacks signed 8-bit In-Phase & Quadrature (I/Q) subcarrier bytes
- Formats structured JSON/Binary payload and pushes to Kafka topic 'rf.esp32.csi.raw'
=============================================================================
"""

import time
import json
import argparse
import logging
import serial
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SerialBridge")


def parse_csi_line(line: str) -> dict:
    """
    Parses a single line of raw CSV data from the ESP32 receiver.ino:
    Format: CSI_DATA,rssi,noise_floor,rate,len,byte0,byte1,...
    """
    tokens = line.strip().split(",")
    if len(tokens) < 5 or tokens[0] != "CSI_DATA":
        return None

    rssi = int(tokens[1])
    noise_floor = int(tokens[2])
    rate = int(tokens[3])
    data_len = int(tokens[4])
    raw_bytes = [int(x) for x in tokens[5:] if x.strip()]

    # ESP32 CSI data buffer contains alternating imaginary & real subcarrier components
    # data_len bytes = (data_len / 2) complex subcarrier values
    num_subcarriers = len(raw_bytes) // 2
    if num_subcarriers == 0:
        return None

    raw_arr = np.array(raw_bytes[: num_subcarriers * 2], dtype=np.float32)
    # Shape: [Subcarriers, 2] -> [Real, Imag]
    # Note: ESP32 standard layout is (imag, real) or (real, imag) depending on core SDK
    imag_part = raw_arr[0::2]
    real_part = raw_arr[1::2]
    
    # Shape for 1 antenna: [1, Subcarriers, 2]
    csi_matrix = np.stack([real_part, imag_part], axis=-1)[np.newaxis, ...].tolist()

    return {
        "timestamp": time.time(),
        "rssi": rssi,
        "noise_floor": noise_floor,
        "rate": rate,
        "num_antennas": 1,
        "num_subcarriers": num_subcarriers,
        "csi_matrix": csi_matrix
    }


def run_bridge(
    port: str = "COM3",
    baudrate: int = 115200,
    kafka_servers: str = "localhost:9092",
    topic: str = "rf.esp32.csi.raw"
):
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        logger.info(f"Connected to Kafka broker at {kafka_servers}. Topic: '{topic}'")
    except Exception as e:
        logger.warning(f"Could not connect to Kafka ({e}). Running in log-only mode.")
        producer = None

    logger.info(f"Opening Serial Port '{port}' at {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
    except Exception as e:
        logger.error(f"Failed to open serial port {port}: {e}")
        return

    packet_count = 0
    logger.info("Serial bridge active. Listening for incoming CSI packets from ESP32...")

    try:
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line or not raw_line.startswith("CSI_DATA"):
                continue

            parsed = parse_csi_line(raw_line)
            if parsed is None:
                continue

            if producer:
                producer.send(topic, value=parsed)

            packet_count += 1
            if packet_count % 100 == 0:
                logger.info(f"Forwarded {packet_count} packets | RSSI: {parsed['rssi']} dBm | Subcarriers: {parsed['num_subcarriers']}")

    except KeyboardInterrupt:
        logger.info("Bridge stopped by user.")
    finally:
        ser.close()
        if producer:
            producer.flush()
            producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32 Serial-to-Kafka CSI Bridge")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", default="rf.esp32.csi.raw", help="Kafka topic name")
    args = parser.parse_args()

    run_bridge(port=args.port, baudrate=args.baud, kafka_servers=args.kafka, topic=args.topic)
