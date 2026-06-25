import cv2
import math
from ultralytics import YOLO

# Load your trained model
model = YOLO("yolov8n.pt")


BALL_CLASS_ID = 32  # Change this to your football class ID
PLAYER_CLASS_ID = 0
POSSESSION_DISTANCE_FACTOR = 1.2

pass_count = 0
last_possession_player = None


def box_center(box):
    x1, y1, x2, y2 = map(float, box.xyxy[0])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box):
    x1, y1, x2, y2 = map(float, box.xyxy[0])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def draw_label(frame, text, x, y, color):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

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
        conf=0.25,
        classes=[PLAYER_CLASS_ID, BALL_CLASS_ID],
        verbose=False
    )

    ball_box = None
    ball_track_id = -1
    player_candidates = []

    # Process detections
    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            # Get class ID
            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if cls == BALL_CLASS_ID:
                ball_box = (x1, y1, x2, y2)
                ball_track_id = track_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_label(frame, f"Ball ID:{track_id}", x1, y1 - 10, (0, 255, 0))

            elif cls == PLAYER_CLASS_ID:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
                draw_label(frame, f"Player ID:{track_id}", x1, y1 - 10, (255, 140, 0))
                player_candidates.append((track_id, box))

    possession_text = "Possession: unknown"
    current_possession_player = None

    if ball_box is not None and player_candidates:
        ball_center_x = (ball_box[0] + ball_box[2]) / 2.0
        ball_center_y = (ball_box[1] + ball_box[3]) / 2.0
        best_player = None
        best_player_box = None
        best_distance = float("inf")

        for track_id, player_box in player_candidates:
            player_center_x, player_center_y = box_center(player_box)
            distance = math.hypot(ball_center_x - player_center_x, ball_center_y - player_center_y)
            if distance < best_distance:
                best_distance = distance
                best_player = track_id
                best_player_box = player_box

        player_scale = max(1.0, math.sqrt(box_area(best_player_box))) if best_player_box is not None else 1.0
        max_allowed_distance = max(60.0, player_scale * POSSESSION_DISTANCE_FACTOR)

        if best_player is not None and best_distance <= max_allowed_distance:
            current_possession_player = best_player
            possession_text = f"Possession: Player {best_player}"

            if best_player_box is not None:
                player_center_x, player_center_y = box_center(best_player_box)
                cv2.line(
                    frame,
                    (int(ball_center_x), int(ball_center_y)),
                    (int(player_center_x), int(player_center_y)),
                    (0, 255, 255),
                    2,
                )

    if current_possession_player is not None and last_possession_player is not None:
        if current_possession_player != last_possession_player:
            pass_count += 1

    if current_possession_player is not None:
        last_possession_player = current_possession_player

    draw_label(frame, possession_text, 20, 40, (0, 255, 255))
    draw_label(frame, f"Passes: {pass_count}", 20, 70, (0, 255, 255))
                
    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Interrupted by user.")
        break
cap.release()
cv2.destroyAllWindows()
