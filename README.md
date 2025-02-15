# Hand Gesture Recognition & Control

<table>
  <tr>
    <td>
      <h3>Pose Tracking/INPUT</h3>
      <img src="Continuous3DHandPoseTrackingusingMachineLearningonline-video-cutter.com-ezgif.com-video-to-gif-converter.gif" width="50%" max-width="800">
    </td>
     <td>
      <h3>Simulation Video/OUTPUT</h3>
      <img src="Output.gif" width="50%" max-width="800">
    </td>
  </tr>
</table>

## Overview

Hand Gesture Recognition & Control is an advanced deep learning-based system designed to recognize and interpret human hand gestures in real time. This project is a combination of Computer Vision, Machine Learning, and AI-driven control systems. The core functionality revolves around using hand gestures as inputs to control various devices, such as drones, smart appliances, and even robotic arms.

The project integrates multiple AI frameworks and technologies, including:

- **Convolutional Neural Networks (CNNs)** for feature extraction from images
- **Recurrent Neural Networks (LSTMs)** for sequential gesture recognition
- **MediaPipe and OpenCV** for real-time hand tracking
- **YOLO-based Object Detection** for gesture classification
- **V-REP Simulation** to visualize gesture-based drone control
- **ONNX Model Optimization** for deployment on edge devices
- This project aims to bridge the gap between human-computer interaction by enabling users to control devices using just their hand gestures, without needing physical controllers.

## Features

- **Real-Time Hand Gesture Recognition:** Utilizes deep learning models to classify gestures instantly.
- **AI-Powered Gesture-Controlled Drone Simulation:** Demonstrates gesture-based flight control in V-REP.
- **Dataset Collection & Custom Gesture Training:** Allows users to train models on their custom gestures.
- **Seamless Integration with Edge Devices:** Optimized for real-time inference on devices like Raspberry Pi.
- **Multi-Modal Fusion for AI Interaction:** Future scope includes voice integration for enhanced interaction.


---

## Installation

### Prerequisites

Ensure that you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

You may also need to install additional dependencies based on the modules you use.

### Clone the Repository

```bash
git clone https://github.com/your-repo/Hand-Gesture-Recognition-Control.git
cd Hand-Gesture-Recognition-Control
```

---

## Dataset

The dataset used for training consists of multiple hand gesture images labeled for different actions. Below are some of the sample gestures used in training:

### Hand Gesture Dataset Samples

![Dataset Sample 1](Hand-gestures-in-our-dataset.png)
![Dataset Sample 2](Hand-gesture-dataset-collected-for-training-and-test-dataset.png)

### Gesture Classes

![Gesture Classes](classes_gestures.png)

---

## Model Architecture

The architecture is built using a combination of:
- Convolutional Neural Networks (CNNs) for feature extraction
- LSTMs for sequence-based hand movement recognition
- MediaPipe framework for robust hand tracking
- ONNX models for optimized inference

### Hand Landmarks Detection

![Hand Landmarks](hand-landmarks.png)

---

## Training Process

The training process involves data augmentation, model optimization, and validation:

```bash
python keypoint_classification.ipynb
python point_history_classification.ipynb
```

### Training Pipeline Visualization

![Training Graph](2-Figure1-1.png)

---

## Inference and Real-Time Gesture Recognition

To run the inference:

```bash
python app.py
```

This will start real-time gesture recognition using the webcam.

### Example of Real-Time Hand Gesture Detection

![Real-Time Gesture Detection](Screenshot.png)

### Hand Gesture Recognition in Action

![Hand Gesture Recognition](10-Figure12-1.png)

---

## Gesture-Controlled Drone Simulation

This project also includes a simulated drone that can be controlled using recognized hand gestures. The simulation is implemented in V-REP.

### Simulation Video

![Drone Simulation](Output.gif)

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

- **Multi-Modal Fusion**: Integration with voice commands for enhanced control
- **Optimized Deployment**: Using TensorRT for efficient inference
- **Gesture Customization**: Allowing users to add custom gestures for personalized interaction

---

## Conclusion

This project serves as a robust foundation for real-time hand gesture recognition and AI-controlled applications. With future improvements, this system can be used in various domains, including AR/VR, robotics, and smart environments.

For any questions or contributions, feel free to open an issue or a pull request!

