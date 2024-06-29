"""
Author: jhzhu
Date: 2024/6/29
Description: 
"""
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
# print(model)
results = model('../../dataset/animal.jpg')
img = results[0].plot()
cv2.imshow('animal', img)
cv2.waitKey()
