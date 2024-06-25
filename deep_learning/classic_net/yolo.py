"""
Author: jhzhu
Date: 2024/6/22
Description: 
"""
from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights

# Train the model
model.train(data="../../dataset/gestures", epochs=100, imgsz=64, save_dir='./yolov8n-cls')
model = YOLO('yolov8n-cls/best.pt')
img_path = '../../dataset/gestures/test/G0/IMG_1312.JPG.jpg'
img = Image.open(img_path)
results = model.predict(img)
print(results)