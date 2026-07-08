import cv2
from ultralytics import YOLO

# Load AI model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run AI detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("AI Vision", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()