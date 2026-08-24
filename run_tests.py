"""
run_tests.py
Universal test runner for the RF-Pose3D Wi-Fi Sensing system.
Works regardless of current working directory.
"""
import os
import sys
import unittest

# Find the tests directory
script_dir = os.path.dirname(os.path.abspath(__file__))
possible_dirs = [
    os.path.join(script_dir, "Wi-Fi-Sensing-ESP32-", "analytics", "tests"),
    os.path.join(script_dir, "analytics", "tests"),
    os.path.join(script_dir, "tests")
]

test_dir = None
for p in possible_dirs:
    if os.path.isdir(p):
        test_dir = p
        break

if not test_dir:
    print(f"Error: Could not locate analytics/tests directory from {script_dir}")
    sys.exit(1)

analytics_dir = os.path.abspath(os.path.join(test_dir, ".."))
if analytics_dir not in sys.path:
    sys.path.insert(0, analytics_dir)

loader = unittest.TestLoader()
suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
