# Hand Gesture Recognition & Control

## Overview

Hand Gesture Recognition & Control is an advanced AI-powered system that enables real-time recognition and interpretation of hand gestures for various applications such as gesture-controlled drones, AI-based human-computer interaction, and assistive technologies. This project integrates multiple deep learning techniques, including Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTMs), MediaPipe, YOLO, and ONNX, to achieve robust gesture recognition and control.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Inference & Real-Time Gesture Recognition](#inference-and-real-time-gesture-recognition)
- [Gesture-Controlled Drone Simulation](#gesture-controlled-drone-simulation)
- [Real-Time Hand Pose Tracking](#real-time-hand-pose-tracking)
- [Edge Deployment](#edge-deployment)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## Features

### 🔹 Real-Time Hand Gesture Recognition
- Utilizes CNN-based models and MediaPipe for robust hand tracking.
- Supports multiple hand gestures with precise landmark detection.

### 🔹 Gesture-Controlled Drone Simulation
- Enables drone control using recognized hand gestures in a simulated environment.
- Implemented using V-REP.

### 🔹 Dataset Collection and Training Pipeline
- Includes a diverse dataset of hand gestures for training.
- Covers various angles, lighting conditions, and backgrounds.

### 🔹 Edge Device Compatibility
- Optimized for deployment on embedded devices like Raspberry Pi and Jetson Nano.
- Supports ONNX model conversion for efficient inference.

### 🔹 Multi-Modal AI Fusion
- Future integration plans with voice and facial recognition.

---

## Installation

### Prerequisites
Ensure that you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

You may also need additional dependencies depending on the platform.

### Clone the Repository
```bash
git clone https://github.com/your-repo/Hand-Gesture-Recognition-Control.git
cd Hand-Gesture-Recognition-Control
```

---

## Dataset

The dataset used for training consists of multiple hand gesture images labeled for different actions.

### Hand Gesture Dataset Samples

![Dataset Sample 1](hand-gestures-in-our-dataset.png)
![Dataset Sample 2](hand-gesture-dataset-collected-for-training-and-test-dataset.png)

### Gesture Classes

![Gesture Classes](classes_gestures.png)

---

## Model Architecture

The system architecture consists of:
- **Convolutional Neural Networks (CNNs):** Extracts spatial features from hand images.
- **LSTMs:** Processes sequences of hand movements for gesture recognition.
- **MediaPipe:** Provides real-time hand tracking and landmark detection.
- **ONNX Models:** Optimized models for deployment on edge devices.

### Hand Landmarks Detection

![Hand Landmarks](hand-landmarks.png)

---

## Training Process

The training pipeline includes dataset preprocessing, augmentation, model training, and evaluation.

```bash
python keypoint_classification.ipynb
python point_history_classification.ipynb
```

### Training Visualization

![Training Graph](2-Figure1-1.png)

---

## Inference and Real-Time Gesture Recognition

To run the inference:
```bash
python app.py
```

### Real-Time Gesture Detection

![Real-Time Gesture Detection](Screenshot.png)

### Hand Gesture Recognition in Action

![Hand Gesture Recognition](10-Figure12-1.png)

---

## Gesture-Controlled Drone Simulation

This project includes a simulated drone that can be controlled using hand gestures. The simulation is implemented in V-REP.

### Simulation Video

![Drone Simulation](Gesture-controlled-drone-simulation.mp4)

### Drone Simulation Framework

![Drone Framework](17-Figure23-1.png)

---

## Real-Time Hand Pose Tracking

A continuous hand tracking system allows gesture-based AI interaction.

### Pose Tracking GIF

![Pose Tracking GIF](Continuous3DHandPoseTrackingusingMachineLearningonline-video-cutter.com-ezgif.com-video-to-gif-converter.gif)

---

## Edge Deployment

This system supports deployment on edge devices such as Raspberry Pi or other embedded systems for real-time low-power inference.

### Edge Deployment Image

![Edge Deployment](accel_multicore_data_capture.png)

---

## Future Improvements

- **Multi-Modal Fusion:** Integrate voice and gesture recognition for better interaction.
- **Optimized Deployment:** Convert models using TensorRT for improved efficiency.
- **Customizable Gestures:** Enable users to define and train their own gesture sets.

---

## Conclusion

Hand Gesture Recognition & Control is a robust foundation for real-time gesture recognition and AI-controlled applications. With future enhancements, it can be applied in robotics, AR/VR, smart environments, and assistive technologies. Contributions and feedback are welcome!

For any questions or contributions, feel free to open an issue or a pull request!

