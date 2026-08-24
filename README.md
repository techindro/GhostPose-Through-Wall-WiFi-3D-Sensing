# RF-Sense3D: Edge-Scalable Through-the-Wall Wi-Fi Sensing System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20(802.11%20OFDM)-E7352C.svg?logo=espressif&logoColor=white)](https://espressif.com)
[![Three.js](https://img.shields.io/badge/WebGL-Three.js%203D-000000.svg?logo=threedotjs&logoColor=white)](https://threejs.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20ASGI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

A production-grade, hardware-agnostic architecture for **Passive Through-the-Wall Human Sensing**, **3D Skeleton Reconstruction**, and **Contactless Vital Signs Monitoring** using raw Wi-Fi Channel State Information (CSI) or RF raw payloads, completely bypassing the need for wearable sensors or optical cameras.

> *Inspired by the foundational RF-Pose & RF-Pose3D research principles (Zhao et al., MIT CSAIL).*

![RF-Sense3D Holographic Web Dashboard](assets/images/rf_pose_dashboard_ui.jpg)

---

## 🛠️ Complete Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RF-SENSE3D ECOSYSTEM                          │
├───────────────────┬────────────────────────────┬────────────────────────┤
│  EMBEDDED & PHY   │   DSP & DEEP LEARNING AI   │   BACKEND & 3D UI/UX   │
├───────────────────┼────────────────────────────┼────────────────────────┤
│ • C++ / Arduino   │ • PyTorch (CUDA FP16)      │ • FastAPI & WebSockets │
│ • ESP-IDF CSI API │ • SciPy & NumPy DSP        │ • Three.js (WebGL 3D)  │
│ • 802.11 OFDM PHY │ • Spatio-Temporal 2D+1D    │ • Apache Kafka Streams │
│ • PySerial Bridge │ • Temporal Self-Attention  │ • Tailwind Glassmorphic│
│ • 921.6k Baud DMA │ • ArcFace Re-ID Learning   │ • Docker & Kubernetes  │
│ • 64 Subcarriers  │ • 3D Kalman Multi-Tracking │ • Supabase Telemetry   │
└───────────────────┴────────────────────────────┴────────────────────────┘
```

### 1. Embedded Firmware & Physical Layer (PHY)
* **C++ / Arduino Core for ESP32**: High-performance firmware running promiscuous Wi-Fi frame sniffing and packet injection.
* **ESP-IDF CSI API (`esp_wifi_set_csi_rx_cb`)**: Real-time extraction of raw 64 OFDM subcarrier complex $I/Q$ channel matrices.
* **DMA Ring Buffering & Serial Streaming**: Custom high-throughput async gateway streaming at **921,600 baud** over USB/UART without packet drops.

### 2. Digital Signal Processing (DSP) & Phase Calibration
* **Linear Phase Sanitization**: Removes Subcarrier Frequency Offset (SFO) linear phase tilt and Carrier Frequency Offset (CFO) random intercept via least-squares phase unwrapping.
* **Recursive EMA Static Clutter Removal**: Exponential moving average filtering to eliminate static multipath reflections from concrete walls and furniture.
* **Dual Butterworth IIR Filter Banks**:
  * **Locomotion Filter**: $0.5 - 5.0\,\text{Hz}$ (Walking gaits, limbs, torso gestures).
  * **Respiration Filter**: $0.1 - 0.45\,\text{Hz}$ (Contactless chest wall expansion: $6 - 27\,\text{BrPM}$).
  * **Cardiac Filter**: $0.8 - 2.5\,\text{Hz}$ (Micro-Doppler heart rate harmonics: $48 - 150\,\text{BPM}$).
* **1D/2D FFT & Doppler Spectrograms**: Real-time extraction of micro-Doppler velocity shifts and spectrogram energy waterfalls.

### 3. Deep Learning & Kinematic AI Architecture
* **PyTorch (CUDA Accelerated)**: GPU FP16 mixed-precision tensor processing engine running at **100+ FPS**.
* **Spatio-Temporal Conv2D+1D Residual Network**: Factorized spatial and temporal convolutional blocks preserving spatial subcarrier correlations and motion temporal dynamics.
* **Temporal Self-Attention Transformers**: Multi-head self-attention layers to model long-range human movement trajectories and complex body interactions.
* **ArcFace Re-Identification Metric Learning**: Angular margin loss generating normalized 128-dimensional biometric embeddings to prevent target ID swapping during multi-person crossings.
* **Kinematic Multi-Task Loss**: Smooth L1 3D pose loss with **anatomical bone-length geometric consistency constraints** to prevent limb warping.
* **Cross-Modal Vision Distillation (Teacher-Student)**: 3D ResNet/OpenPose teacher network distilling visual supervision into the RF student network.

### 4. Multi-Target 3D Tracking & Spatial ROI
* **3D Constant-Velocity Kalman Filtering**: State estimation predicting target 3D bounding boxes and velocity vectors.
* **Hungarian / Munkres Algorithm**: Global bipartite matching combining Euclidean spatial distance with ArcFace cosine affinity.
* **Spatial ROI & Multipath Ghost Filter**: Axis-Aligned Bounding Box (AABB) room boundary filtering to discard out-of-boundary multipath ghost reflections.

### 5. Backend Microservices & Streaming Data Gateway
* **FastAPI (Async Python ASGI)**: Microsecond-latency REST and WebSocket gateway.
* **High-Frequency WebSocket Engine**: Real-time bi-directional binary/JSON pipeline streaming 3D keypoints, vitals, and latency telemetry to connected dashboards.
* **Apache Kafka & Zookeeper (`aiokafka`)**: Distributed event streaming platform buffering high-rate CSI packet topics (`rf.esp32.csi.raw`).
* **OAuth 2.0 & Bearer JWT Security**: Role-based access control (RBAC) with cryptographic token validation.
* **Supabase & Cloud Telemetry**: PostgreSQL time-series store for long-term health metrics logging.

### 6. Frontend 3D Engine & Holographic UI/UX
* **Three.js (WebGL)**: Real-time 3D rendering of volumetric human avatars, glowing joint nodes, neon bone cylinders, bounding boxes, and coordinate laser grid floors.
* **HTML5 Canvas 2D API**: High-precision 120 Hz chest wall displacement and ECG respiration waveforms.
* **Tailwind CSS & Glassmorphism**: Futuristic dark HUD theme (`#030712`) with frosted blur backdrops, neon cyan/rose accents, and animated circular radar sweep scope.
* **OrbitControls**: 360° orbital spatial rotation, pan, and smooth zoom controls.
* **Typography**: Google Fonts (*Orbitron*, *Space Grotesk*, *JetBrains Mono*).

### 7. DevOps, Cloud & Containerization
* **Docker & Multi-Stage Builds**: Containerized inference services with CUDA runtime drivers.
* **Docker Compose**: One-click orchestration of Kafka, Zookeeper, and Inference Gateway.
* **Kubernetes (`k8s/rf-pose-deployment.yaml`)**: Scalable deployment manifests with `nvidia.com/gpu: 1` resource limits and health probes.

---

## 🔬 System Architecture & Physical Pipeline

![RF-Sense3D Technical Architecture](assets/images/rf_pose_architecture_diagram.jpg)

The system operates across three fundamental layers:
1. **RF Illumination & Physical Propagation**: 2.4 GHz OFDM Wi-Fi radio frequency waves penetrate physical obstacles (concrete walls, partitions, pitch darkness) and reflect off human bodies.
2. **RF Signal Reception & Phase Calibration**: Multi-antenna receiver arrays capture raw Channel State Information (CSI). Subcarrier phase linear sanitization untwists SFO and cancels CFO offsets while Butterworth filter banks isolate cardiopulmonary micro-Doppler signals.
3. **Deep Learning Inference & 3D Holographic Rendering**: A Spatio-Temporal PyTorch neural network (`RFStudentNetwork`) estimates 17 3D skeletal keypoints, calculates ArcFace Re-ID embeddings, and extracts respiration/heart rate waveforms streaming at 100 FPS into an interactive Three.js 3D Web UI.

---

## 🌐 Research References & Public Wi-Fi Datasets

If you want to explore the foundational research literature, video demonstrations, or benchmark on public datasets:

| Project / Dataset | Official Link | Description |
|---|---|---|
| **MIT CSAIL RF-Pose Research** | [http://rfpose.csail.mit.edu/](http://rfpose.csail.mit.edu/) | Landmark research paper on through-wall human pose estimation (CVPR). |
| **MIT RF-Pose3D Research** | [http://rfpose3d.csail.mit.edu/](http://rfpose3d.csail.mit.edu/) | 3D human pose reconstruction literature from radio frequency signals (SIGCOMM). |
| **Tsinghua University Widar 3.0** | [http://tns.thss.tsinghua.edu.cn/widar3.0/](http://tns.thss.tsinghua.edu.cn/widar3.0/) | Large open-source Wi-Fi CSI human sensing dataset for walking gaits and tracking. |
| **Espressif Official ESP-CSI** | [https://github.com/espressif/esp-csi](https://github.com/espressif/esp-csi) | Official Espressif framework and dataset for human presence detection using ESP32. |
| **CSI-Bench Benchmark Suite** | [https://github.com/geek-ai/csi-bench](https://github.com/geek-ai/csi-bench) | Deep learning benchmarking suite on raw Wi-Fi CSI data. |

---

## Key Capabilities

1. **Through-Wall 3D Skeleton Estimation**: Reconstructs 17 COCO 3D kinematic human stick figures and volumetric avatars in real-time behind concrete walls.
2. **Device-Free Multi-Person Tracking & Re-ID**: Simultaneously tracks multiple distinct human targets (Subject #1 & Subject #2), preventing identity swaps with ArcFace metric learning and 3D Kalman filtering.
3. **Real-Time Emergency Fall Detection & Dispatcher**: Bio-kinematic state machine calculating vertical velocity drop ($\frac{dz}{dt} < -1.4\,\text{m/s}$) and bounding-box aspect ratio inversion to trigger immediate visual strobe and audio alarms within $150\,\text{ms}$.
4. **Contactless Vital Signs Monitoring**: Micro-Doppler phase shift analysis to compute respiration rate (BrPM) and heart rate variations (BPM).
5. **Micro-Doppler Time-Frequency Spectrogram**: Animated energy waterfall tracking human locomotion Doppler frequencies ($\pm 15\,\text{Hz}$).
6. **Lighting & NLoS Invariance**: Fully functional in total darkness, smoke, and non-line-of-sight environments.
7. **Cross-Modal Teacher-Student Architecture**: Supervised by vision teacher networks (OpenPose/DensePose) with knowledge distillation into a Spatio-Temporal RF Student Network.

---

## Directory Structure

```
Wi-Fi-Sensing-ESP32-/
├── assets/
│   └── images/                      # Architecture diagrams & UI screenshots
├── main.py                          # Universal CLI controller (Dashboard, Training, Tests)
├── run_tests.py                     # Universal test runner
├── start.bat                        # One-click Windows dashboard launcher
├── analytics/
│   ├── dashboard/
│   │   └── index.html               # Three.js 3D Skeleton & Vitals Dashboard
│   ├── tests/
│   │   └── test_rf_pipeline.py      # Unit & Latency Benchmark Tests
│   ├── rf_signal_processor.py       # DSP, Phase Sanitization & Filter Banks
│   ├── rf_pose_model.py             # PyTorch RFStudentNetwork & Multi-Task Loss
│   ├── multi_target_tracker.py      # 3D Kalman Filter + Hungarian Re-ID Anti-Swap Tracker
│   ├── vision_teacher_network.py    # Cross-Modal 3D ResNet/OpenPose Vision Teacher & KD Loss
│   ├── roi_masking.py               # 3D Spatial Region-of-Interest & Multipath Ghost Filter
│   ├── auth_security.py             # OAuth 2.0 Bearer JWT Authentication & Captcha Verification
│   ├── analytics_store.py           # Supabase & Time-Series Telemetry Persistence Gateway
│   ├── train_rf_pose.py             # Deep Learning Model Training & Evaluation Engine
│   ├── train_model.py               # Unified Trainer (Deep Learning + Classical Presence)
│   ├── kafka_inference_service.py   # Async FastAPI & Kafka/WebSocket Server
│   ├── rf_stream_simulator.py       # Synthetic RF-CSI Physics Generator
│   ├── serial_to_kafka_bridge.py    # ESP32 Serial to Kafka/WebSocket Bridge
│   ├── data_collector.py            # Dataset Recording CLI
│   └── requirements.txt             # Python Dependencies
├── firmware/
│   ├── transmitter/
│   └── receiver/
├── k8s/
│   └── rf-pose-deployment.yaml      # Kubernetes GPU Deployment Manifest (nvidia.com/gpu: 1)
├── Dockerfile                       # GPU Container Definition
├── docker-compose.yml               # Kafka, Zookeeper & Inference Orchestration
└── LICENSE                          # MIT Open Source License
```

---

## Quickstart Guide

### 1. Installation
```bash
pip install -r Wi-Fi-Sensing-ESP32-/analytics/requirements.txt
```

### 2. Run the Real-Time 3D Sensing Engine & Web Dashboard
```bash
python main.py --dashboard
```
Or simply double-click **`start.bat`**.
Open your browser at **`http://localhost:8000`** to interact with the live 3D skeleton visualizer, breathing waveforms, and Re-ID tracks.

### 3. Run Automated Tests & Latency Benchmark
```bash
python run_tests.py
```

### 4. Train the Model with Kinematic & Vital Loss
```bash
python main.py --train
```

### 5. Stream Real ESP32 Hardware
Flash `firmware/transmitter/transmitter.ino` and `firmware/receiver/receiver.ino` to your two ESP32 devices. Then start the serial gateway:
```bash
python main.py --bridge --serial-port COM3
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 techindro.
