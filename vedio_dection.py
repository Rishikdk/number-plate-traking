# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load the trained YOLOv8 model
# model = YOLO('D:/Python/traking/runs/detect/train/weights/best.pt')

# # Open the video file
# video_capture = cv2.VideoCapture("test.mp4")

# # Define codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# video_writer = cv2.VideoWriter('Output/vedio/output_video.mp4', fourcc, fps, (width, height))

# # Set the delay between frames (in milliseconds)
# frame_delay = int(500 / fps) * 2  # For example, doubling the frame delay

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Perform inference on the current frame
#     results = model(frame)

#     # Process results
#     detections = results[0].boxes  # Accessing the first item in results, which contains the bounding boxes

#     # Draw bounding boxes and labels
#     for detection in detections:
#         x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)  # Convert tensor to numpy array and then to int
#         label = int(detection.cls[0])  # Class label (as index)
#         confidence = detection.conf[0]  # Confidence score
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame with bounding boxes
#     cv2.imshow('Video', frame)

#     # Write the frame with bounding boxes to the output video
#     video_writer.write(frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
#         break

# # Release resources
# video_capture.release()
# video_writer.release()
# cv2.destroyAllWindows()


from ultralytics import RTDETR

# Load a model
model = RTDETR("D:/Python/traking/runs/detect/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model(source="test.mp4" , show=True, conf=0.4, save=True , stream=True)  # predict on an image