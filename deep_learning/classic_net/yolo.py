"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.yaml")  # build a new model from YAML
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="gestures", epochs=100, imgsz=64)
