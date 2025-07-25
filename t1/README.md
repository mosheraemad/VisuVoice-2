# 🚗 Road Signs Detection using YOLOv8

This project demonstrates how to train and evaluate a YOLOv8 object detection model for road signs using a dataset from Kaggle. The entire workflow is built and executed in Google Colab.

## 📌 Project Overview

- Download dataset from Kaggle
- Visualize sample images
- Train YOLOv8 on road sign detection
- Evaluate the model and plot metrics
- Predict on test images and visualize results
- Export model to ONNX
- Save and download training outputs

---

## 📂 Dataset

- Dataset: [`pkdarabi/cardetection`](https://www.kaggle.com/datasets/pkdarabi/cardetection)
- Contains images and labels of 15 different road sign categories.

---

## 🧠 Model

- YOLOv8n (Nano) model from Ultralytics
- Trained for 30 epochs using the provided data split
- Exported to ONNX for deployment

---

## 🛠️ Installation

```bash
pip install ultralytics kaggle
