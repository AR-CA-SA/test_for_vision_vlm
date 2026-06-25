import cv2
from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo11n.pt")

# Open the video file using OpenCV
cap = cv2.VideoCapture("test_video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO26 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO26 Inference", annotated_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()