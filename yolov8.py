# from ultralytics import YOLO
# import cv2

# # Load the trained YOLOv8 model
# model = YOLO('D:\Python/traking/runs/detect/train/weights/best.pt')

# # Read the image
# image = cv2.imread('A-391-_jpg.rf.0664c52bfa80a3bba75435b5d8804e95.jpg')

# # Perform inference
# results = model(image)

# # Process results
# # Access results as a list of detections
# detections = results[0].boxes  # Accessing the first item in results, which contains the bounding boxes

# # Draw bounding boxes and labels
# for detection in detections:
#     x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
#     label = detection.cls[0]  # Class label (as index)
#     confidence = detection.conf[0]  # Confidence score
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# # Save or display the output image
# cv2.imwrite('Output/img/output_image.jpg', image)
# # cv2.imshow('Result', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

from ultralytics import YOLO

# Load a model
model = YOLO("D:\Python/traking/runs/detect/train/weights/best.pt")  # load a custom trained model

# Export the model
results = model(source="0", show=True, conf=0.4, save=True)