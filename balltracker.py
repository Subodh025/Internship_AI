import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("yolov8n.pt")


BALL_CLASS_ID = 32  # Change this to your football class ID

# Open video
cap = cv2.VideoCapture("kdb.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO + BoT-SORT
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.3,               # Lower confidence to detect small objects like a ball
        classes=[BALL_CLASS_ID], # Tell YOLO to ONLY look for the ball (reduces false positives from other objects)
        verbose=False
    )

    # Process detections
    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            # Get class ID
            cls = int(box.cls[0])

            # Change this to your football class ID
            if cls == BALL_CLASS_ID:

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Tracking ID
                track_id = int(box.id[0]) if box.id is not None else -1

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                cv2.putText(
                    frame,
                    f"Ball ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Interrupted by user.")
        break
cap.release()
cv2.destroyAllWindows()

#this is 2nd branch hope the main branch doesn't change