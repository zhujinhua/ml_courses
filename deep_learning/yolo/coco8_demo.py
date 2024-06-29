"""
Author: jhzhu
Date: 2024/6/29
Description: 
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="coco8.yaml", epochs=10, imgsz=640, workers=1)