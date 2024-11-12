# Tennis-Players-Court-Detection

[![Watch the Video]](https://github.com/YousefEldaly/Tennis-Players-Court-Detection/blob/main/output_videos/output_video.gif)

This repository provides a complete system for detecting players, balls, and court keypoints in tennis footage. It leverages various deep learning models for each detection task, including YOLOv8 for players, a custom-trained YOLOv11 for the tennis ball, and a fine-tuned ResNet50 for court keypoints. Missing detections are interpolated to maintain smooth tracking, and only people near the court are filtered and classified as players.
This is made following this tutorial [Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction] (https://www.youtube.com/watch?v=L23oIHZE14w&t=8993s)

## Key differences

- **Using YOLOv11 instead of YOLOv5**: For tennis ball detection, YOLOv11 was trained on custom data, achieving higher accuracy for this specific task compared to YOLOv5.
  ![Check Performance Comparison](https://github.com/YousefEldaly/Tennis-Players-Court-Detection/blob/main/performance-comparison.png)
- **Resolved Issues with the original repo**: An issue was encountered and reported in the original [tennis_analysis](https://github.com/abdullahtarek/tennis_analysis/issues/5)project repository regarding an error in the training script. The issue was successfully resolved.

---

## System Overview

The system consists of three main components:

1. **Player Detection**: Detects people near the court to identify players using YOLOv8.
2. **Ball Detection**: Custom-trained YOLOv11 model detects the tennis ball.
3. **Court Keypoints Detection**: A fine-tuned ResNet50 model with a custom output layer detects 14 keypoints of the court in (x, y) coordinates.

---

## Model Architectures

### YOLOv8 - Player Detection

For player detection, we use [YOLOv8](https://github.com/ultralytics/ultralytics) to accurately detect people in the frame. The detected people are filtered to include only those near the court, identifying them as players.

- **Input Size**: 640x640 pixels
- **mAP (50-95)**: Achieved ~50% mAP
- **Speed**: T4 TensorRT 11.3 ms, CPU ONNX 462.8 ms
- **Parameters**: 56.9M
- **FLOPs**: 194.9B

### YOLOv11 - Tennis Ball Detection

We trained [YOLOv11](https://docs.ultralytics.com/models/yolo11/#overview) on custom data to detect tennis balls with high accuracy.

- **Input Size**: 640x640 pixels
- **mAP (50-95)**: Achieved ~54.7% mAP
- **Speed**: T4 TensorRT 11.3 ms, CPU ONNX 462.8 ms
- **Parameters**: 56.9M
- **FLOPs**: 194.9B

### ResNet50 - Court Keypoints Detection

The court keypoints detector is based on [ResNet50](https://arxiv.org/abs/1512.03385), fine-tuned with a custom fully connected layer to output 14 keypoints (each with (x, y) coordinates).
