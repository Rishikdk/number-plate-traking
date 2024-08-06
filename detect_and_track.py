import cv2
from ultralytics import YOLO
import numpy as np
from sort_module import Sort  # Ensure you have sort.py in the same directory or install SORT

# Load the trained YOLOv8 model
model = YOLO('yolov8m.pt')

# Initialize video capture (0 for webcam or provide a video file path)
cap = cv2.VideoCapture(0)  # Change 0 to 'test.mp4' if you want to use the video file

# Initialize SORT tracker
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Prepare detections for the tracker
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf > 0.3:  # Filter out low confidence detections
                detections.append([x1, y1, x2, y2, conf])
    
    # Update tracker
    trackers = tracker.update(np.array(detections))
    
    # Draw tracked objects
    for d in trackers:
        x1, y1, x2, y2, obj_id = map(int, d)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Number Plate Tracking', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
