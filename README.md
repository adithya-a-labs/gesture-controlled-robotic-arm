# 🤖 Gesture Controlled Robotic Arm

A real-time vision-based robotic arm system that translates human hand gestures into physical robot motion, with a synchronized 3D digital twin, live calibration system, and experimental inverse kinematics pipeline.

---

## 📌 Overview

This project demonstrates a complete end-to-end pipeline that integrates **computer vision**, **real-time control systems**, **embedded hardware**, and **3D visualization** into a unified interactive platform.

Using a webcam, the system tracks human arm and hand movements via MediaPipe, converts them into calibrated servo angles, and controls a physical robotic arm through an ESP32 microcontroller. Simultaneously, a live 3D digital twin mirrors the robot’s motion in real time.

The system is designed with **mechanical constraints**, **signal smoothing**, and **modular architecture**, making it stable, extensible, and suitable for experimentation with advanced control techniques.

---

## 🧠 System Architecture


---

## 🔁 System Pipeline Explained

### 1. Vision Processing
- Captures live video using OpenCV
- Uses MediaPipe to extract body and hand landmarks
- Tracks:
  - Shoulder
  - Elbow
  - Wrist
  - Fingers (for gripper control)

---

### 2. Angle Computation
- Converts landmark positions into joint angles:
  - Base rotation (S4) → torso orientation
  - Shoulder (S3) → arm elevation
  - Elbow (S2) → arm bending
  - Gripper (S1) → pinch detection

---

### 3. Calibration Layer (Core Feature)

Instead of directly mapping human angles to servos, the system applies:

- Range mapping (human → servo)
- Offset correction
- Mechanical constraints
- Direction correction

#### Joint Constraints:

| Joint | Servo | Range | Behavior |
|------|------|------|---------|
| Gripper | S1 | 10 – 100 | Open/close with latch |
| Elbow | S2 | 20 – 75 | Higher angle → extends arm |
| Shoulder | S3 | 55 – 85 | Higher angle → lowers arm |
| Base | S4 | Calibrated | Controlled by torso rotation |

---

### 4. Signal Conditioning

To ensure stable motion:

- Low-pass filtering → reduces jitter  
- Deadband → ignores small noise  
- Rate limiting → smooth motion transitions  

---

### 5. Hardware Control (ESP32)

- Receives servo angles via serial communication
- Applies:
  - smoothing
  - speed limiting
- Drives servos at ~50 Hz for smooth motion

---

### 6. Digital Twin (Three.js)

- Real-time 3D visualization of the robotic arm
- Uses **inverse mapping** (servo → human angle)
- Ensures visual motion matches physical behavior
- Supports multiple dashboards:
  - standard visualization
  - forward kinematics
  - inverse kinematics (experimental)

---

## 🎮 Dashboards

| Dashboard | URL | Description |
|----------|-----|------------|
| 3D Digital Twin | `/3d-fk` | Real-time visualization |
| Calibration Panel | `/tune` | Live parameter tuning |
| IK Dashboard | `/3d-ik` | Experimental IK system |

---

## ⚙️ Installation

```bash
git clone https://github.com/adithya-a-labs/gesture-controlled-robotic-arm.git
cd gesture-controlled-robotic-arm
pip install -r requirements.txt
