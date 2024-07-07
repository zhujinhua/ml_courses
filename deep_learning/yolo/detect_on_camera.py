"""
Author: jhzhu
Date: 2024/6/29
Description: 
"""
from PIL import Image
from ultralytics import YOLO
import cv2

# Load a model
# model = YOLO("yolov8n.pt")
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    status, frame = cap.read()
    if not status:
        print('read failed')
        break
    results = model(frame)
    img = results[0].plot()
    cv2.imshow('frame', img)
    # wait esc to exit
    if cv2.waitKey(int(1000/24)) == 27:
        break

cap.release()
cv2.destroyAllWindows()