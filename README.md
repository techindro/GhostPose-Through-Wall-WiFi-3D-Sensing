# Wi-Fi--Sensing-ESP32-

# Wi-Fi Sensing & Motion Detection using ESP32 CSI

This project demonstrates how to use Wi-fi Channel State Information (CSI) from cheap ESP32 boards to detect human presence and motion behinds without any cameras, inspired by MIT's RF-Pose concept.

## Hardware Required 

- 2x ESP32 Development Boards (WROOM-32)
- 2x Micro-USB/Type-C cabels

## How it works

1. **transmitter ESP32** sends continuous Wi-Fi packets.
2. **Receiver ESP32** captures the packets and extracts the raw CSI data.
3. **Python Script** collects this data via Serial port and trains a Machine Learning model (SVM) to classify "Empty Room" vs "Human Present".
