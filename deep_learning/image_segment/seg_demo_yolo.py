"""
Author: jhzhu
Date: 2024/7/7
Description: 
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="coco8-seg.yaml", epochs=10, imgsz=640)
