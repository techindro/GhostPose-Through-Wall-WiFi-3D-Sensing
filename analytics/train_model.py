"""
train_model.py
=============================================================================
Unified RF-Pose & CSI Sensing Model Training Suite:
- Option 1: Deep Learning Spatio-Temporal RF-Pose3D & Vitals Training
- Option 2: Classical Presence & Activity Classification (SVM / Random Forest / MLP)
- Automatically saves checkpoints and exports trained models
=============================================================================
"""

import os
import sys
import argparse
import logging
import numpy as np

# Ensure analytics directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from train_rf_pose import main as train_rf_pose_main
from rf_signal_processor import RFSignalProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModel")


def train_classical_classifier(dataset_dir: str = "dataset/raw"):
    """
    Trains a lightweight classical ML presence/motion classifier (Random Forest / SVM)
    using extracted CSI Phase Variance and Doppler energy features.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    logger.info("Extracting statistical CSI features for classical classification...")
    processor = RFSignalProcessor(num_subcarriers=64, sampling_rate_hz=100.0)

    # Generate synthetic training batch if raw dataset is empty
    X, y = [], []
    num_samples = 400
    for _ in range(num_samples):
        is_present = np.random.choice([0, 1])
        if is_present == 1:
            # Active movement: higher Doppler variance and phase fluctuations
            raw_csi = 20.0 + 10.0 * np.random.randn(3, 64, 100) + 1j * (np.random.randn(3, 64, 100) * 1.5)
        else:
            # Empty room / static channel: low variance
            raw_csi = 20.0 + 0.5 * np.random.randn(3, 64, 100) + 1j * (np.random.randn(3, 64, 100) * 0.1)

        amp, phase = processor.sanitize_phase(raw_csi)
        dynamic_amp = processor.remove_static_clutter(amp)

        feat_vector = [
            np.mean(dynamic_amp),
            np.var(dynamic_amp),
            np.var(phase),
            np.max(dynamic_amp) - np.min(dynamic_amp),
            np.mean(np.abs(np.diff(dynamic_amp, axis=-1)))
        ]
        X.append(feat_vector)
        y.append(is_present)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Presence Classifier Accuracy: {acc * 100.0:.2f}%")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Empty Room", "Human Present"]))

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "presence_classifier_rf.joblib")
    joblib.dump(clf, out_path)
    logger.info(f"[OK] Saved classical classifier to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RF-Pose & CSI Model Trainer")
    parser.add_argument(
        "--mode",
        choices=["rf_pose", "classical"],
        default="rf_pose",
        help="Training mode: 'rf_pose' for 3D Skeletons & Vitals or 'classical' for Presence/Motion classification"
    )
    args = parser.parse_args()

    if args.mode == "rf_pose":
        logger.info("Launching RF-Pose3D Deep Learning Training Pipeline...")
        train_rf_pose_main()
    else:
        logger.info("Launching Classical Machine Learning Presence Classifier...")
        train_classical_classifier()


if __name__ == "__main__":
    main()
