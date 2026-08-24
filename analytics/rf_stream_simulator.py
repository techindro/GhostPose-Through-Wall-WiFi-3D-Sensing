"""
rf_stream_simulator.py
=============================================================================
High-Fidelity Synthetic CSI Stream Simulator:
- Simulates realistic multi-antenna OFDM Channel State Information (CSI)
- Models human walking, arm swings, chest wall respiration, and heartbeats
- Injects realistic RF Multipath, SFO (Sampling Frequency Offset) linear phase tilt,
  and random Carrier Frequency Offset (CFO)
- Streams generated JSON packets to Kafka topic 'rf.esp32.csi.raw' or stdout
=============================================================================
"""

import time
import json
import argparse
import numpy as np
from typing import Dict, Any


def generate_synthetic_csi_packet(
    t: float,
    num_antennas: int = 3,
    num_subcarriers: int = 64,
    walking_speed: float = 0.8,
    resp_rate_bpm: float = 16.0,
    heart_rate_bpm: float = 72.0
) -> Dict[str, Any]:
    """
    Generates a single complex CSI snapshot array [Antennas, Subcarriers, 2] (Real, Imag).
    Includes multipath, human Doppler reflection, respiration chest vibration,
    plus artificial SFO phase tilt and CFO phase noise.
    """
    # 1. Base Static Channel Multipath
    static_amp = np.random.uniform(20.0, 40.0, size=(num_antennas, num_subcarriers))
    static_phase = np.random.uniform(-np.pi, np.pi, size=(num_antennas, num_subcarriers))
    
    # 2. Dynamic Component: Human Locomotion Doppler Shift
    # Wavelength lambda at 2.4 GHz ~ 0.125m
    doppler_freq = (2.0 * walking_speed) / 0.125  # ~12.8 Hz
    locomotion_mod = 10.0 * np.sin(2 * np.pi * doppler_freq * t)
    
    # 3. Cardiopulmonary Micro-Doppler (Chest wall displacement ~5mm, heart apex ~0.5mm)
    resp_freq = resp_rate_bpm / 60.0
    heart_freq = heart_rate_bpm / 60.0
    vitals_mod = 3.0 * np.sin(2 * np.pi * resp_freq * t) + 0.8 * np.sin(2 * np.pi * heart_freq * t)
    
    # Combined Amplitude
    total_amp = np.maximum(static_amp + locomotion_mod + vitals_mod, 1.0)
    
    # 4. Phase with Injected Imperfections (SFO Linear Tilt + CFO Random Walk)
    subcarrier_idx = np.arange(-num_subcarriers // 2, num_subcarriers // 2)
    sfo_slope = 0.05 * np.sin(2 * np.pi * 0.1 * t)  # Dynamic SFO slope
    cfo_offset = np.random.uniform(-np.pi, np.pi)     # Random CFO offset
    noise = np.random.normal(0, 0.05, size=(num_antennas, num_subcarriers))
    
    total_phase = (
        static_phase +
        (sfo_slope * subcarrier_idx).reshape(1, -1) +
        cfo_offset +
        0.2 * np.sin(2 * np.pi * resp_freq * t) +
        noise
    )
    
    # Convert polar (Amp, Phase) -> Cartesian (Real, Imag)
    real_part = total_amp * np.cos(total_phase)
    imag_part = total_amp * np.sin(total_phase)
    
    # Stack into [Antennas, Subcarriers, 2]
    csi_matrix = np.stack([real_part, imag_part], axis=-1).tolist()
    
    return {
        "timestamp": time.time(),
        "rssi": int(-45 + np.random.randint(-3, 4)),
        "noise_floor": -95,
        "rate": 11,
        "num_antennas": num_antennas,
        "num_subcarriers": num_subcarriers,
        "csi_matrix": csi_matrix
    }


def stream_to_kafka(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "rf.esp32.csi.raw",
    sample_rate_hz: float = 100.0,
    duration_sec: float = 60.0
):
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print(f"[+] Connected to Kafka at {bootstrap_servers}. Streaming to topic '{topic}'...")
    except Exception as e:
        print(f"[-] Could not connect to Kafka ({e}). Running in console emission mode.")
        producer = None

    dt = 1.0 / sample_rate_hz
    start_time = time.time()
    t = 0.0
    packet_count = 0

    print(f"[+] Starting synthetic RF CSI stream at {sample_rate_hz} Hz for {duration_sec}s...")
    
    try:
        while (time.time() - start_time) < duration_sec:
            loop_t0 = time.perf_counter()
            
            packet = generate_synthetic_csi_packet(t)
            if producer:
                producer.send(topic, value=packet)
            
            packet_count += 1
            if packet_count % 100 == 0:
                print(f"  --> Streamed {packet_count} CSI packets | RSSI: {packet['rssi']} dBm | t={t:.2f}s")
                
            t += dt
            elapsed = time.perf_counter() - loop_t0
            sleep_time = max(dt - elapsed, 0.0)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n[!] Stream interrupted by user.")
    finally:
        if producer:
            producer.flush()
            producer.close()
        print(f"[OK] Stream finished. Total packets generated: {packet_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic RF CSI Streamer")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", default="rf.esp32.csi.raw", help="Kafka topic name")
    parser.add_argument("--rate", type=float, default=100.0, help="Sampling rate in Hz")
    parser.add_argument("--duration", type=float, default=120.0, help="Stream duration in seconds")
    args = parser.parse_args()

    stream_to_kafka(
        bootstrap_servers=args.kafka,
        topic=args.topic,
        sample_rate_hz=args.rate,
        duration_sec=args.duration
    )
