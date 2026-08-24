"""
main.py
=============================================================================
Universal Entry Point for MIT RF-Pose & RF-Pose3D Wi-Fi Sensing System:
Usage:
  python main.py --dashboard     (Launches real-time 3D Web Dashboard on http://localhost:8000)
  python main.py --train         (Trains 3D RF-Pose neural network with vitals & kinematics)
  python main.py --test          (Runs automated test suite across all modules)
  python main.py --bridge        (Starts ESP32 Serial-to-Kafka/Inference bridge)
=============================================================================
"""

import os
import sys
import argparse

# Add analytics directory to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_ANALYTICS_DIRS = [
    os.path.join(ROOT_DIR, "Wi-Fi-Sensing-ESP32-", "analytics"),
    os.path.join(ROOT_DIR, "analytics"),
    ROOT_DIR
]

ANALYTICS_DIR = None
for p in POSSIBLE_ANALYTICS_DIRS:
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "rf_pose_model.py")):
        ANALYTICS_DIR = p
        break

if not ANALYTICS_DIR:
    print(f"[!] Error: Could not locate 'analytics' directory from {ROOT_DIR}")
    sys.exit(1)

if ANALYTICS_DIR not in sys.path:
    sys.path.insert(0, ANALYTICS_DIR)


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def run_dashboard(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    os.chdir(ANALYTICS_DIR)

    # Check if target port is already in use and auto-select next available port
    target_port = port
    while is_port_in_use(target_port):
        print(f"[!] Note: Port {target_port} is already in use by a running instance.")
        target_port += 1
        print(f"[+] Automatically switching to available Port {target_port}...")

    print(f"\n[+] Starting MIT RF-Pose3D Real-Time Server on http://localhost:{target_port}")
    print(f"[+] Open your browser at: http://localhost:{target_port} or http://localhost:{target_port}/dashboard\n")
    uvicorn.run("kafka_inference_service:app", host=host, port=target_port, reload=False)


def run_training(mode: str = "rf_pose"):
    import importlib
    os.chdir(ANALYTICS_DIR)
    train_module = importlib.import_module("train_model")
    train_main = getattr(train_module, "main")
    sys.argv = ["train_model.py", "--mode", mode]
    train_main()


def run_tests():
    test_dir = os.path.join(ANALYTICS_DIR, "tests")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def run_bridge(port: str = "COM3", baud: int = 115200):
    import importlib
    os.chdir(ANALYTICS_DIR)
    bridge_module = importlib.import_module("serial_to_kafka_bridge")
    start_bridge = getattr(bridge_module, "run_bridge")
    start_bridge(port=port, baudrate=baud)


def main():
    parser = argparse.ArgumentParser(description="MIT RF-Pose3D Universal Controller")
    parser.add_argument("--dashboard", action="store_true", help="Launch real-time 3D Web Dashboard & Inference Server")
    parser.add_argument("--train", action="store_true", help="Train RF-Pose3D Deep Neural Network")
    parser.add_argument("--train-classical", action="store_true", help="Train Classical Presence/Motion Classifier")
    parser.add_argument("--test", action="store_true", help="Run full automated verification test suite")
    parser.add_argument("--bridge", action="store_true", help="Start ESP32 Serial Hardware Bridge")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--serial-port", default="COM3", help="ESP32 serial port (e.g. COM3)")
    args = parser.parse_args()

    if args.train:
        run_training("rf_pose")
    elif args.train_classical:
        run_training("classical")
    elif args.test:
        run_tests()
    elif args.bridge:
        run_bridge(port=args.serial_port)
    else:
        # Default action: launch dashboard
        run_dashboard(port=args.port)


if __name__ == "__main__":
    main()
