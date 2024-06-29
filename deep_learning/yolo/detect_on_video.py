"""
Author: jhzhu
Date: 2024/6/29
Description: 
"""
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load the model (replace with your specific YOLOv8 model file)

# Open the input video file
input_video_path = '../../dataset/home_video.mp4'  # Replace with your video file path
output_video_path = 'output_video.mp4'  # Output video file path

cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f'{model.names[class_id]} {confidence:.2f}'

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

