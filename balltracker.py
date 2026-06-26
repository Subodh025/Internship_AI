import cv2 
import math 
from ultralytics import YOLO  

# Load your trained model.
model = YOLO("yolov8n.pt")  


BALL_CLASS_ID = 32 
PLAYER_CLASS_ID = 0  
POSSESSION_DISTANCE_FACTOR = 1.2  # Scales the allowed ball-to-player distance for possession.

pass_count = 0  
last_possession_player = None  

def box_center(box):  # Defines a helper that returns the center point of a bounding box.
    x1, y1, x2, y2 = map(float, box.xyxy[0])  # Reads the box corners as floating-point values.
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # Returns the midpoint of the box.


def box_area(box):  # Defines a helper that computes the area of a bounding box.
    x1, y1, x2, y2 = map(float, box.xyxy[0])  # Reads the box corners as floating-point values.
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)  # Returns width times height while preventing negative sizes.


def draw_label(frame, text, x, y, color):  # Defines a helper for drawing text on a video frame.
    cv2.putText(  # Starts an OpenCV text drawing call.
        frame,  # Provides the frame image to draw on.
        text,  # Provides the label text to draw.
        (x, y),  # Sets the top-left text position.
        cv2.FONT_HERSHEY_SIMPLEX,  # Chooses the OpenCV font style.
        0.6,  # Sets the text scale.
        color,  # Sets the text color.
        2,  # Sets the text thickness.
    )  # Finishes drawing the text label.

# Open video.
cap = cv2.VideoCapture("kdb.mp4")

while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break

    # Run YOLO + BoT-SORT.
    results = model.track(  # Runs object detection and tracking on the current frame.
        frame,  # Sends the current video frame into the model.
        persist=True,  # Keeps track identities across consecutive frames.
        tracker="botsort.yaml",  # Uses the BoT-SORT tracker configuration.
        conf=0.25,  # Sets the minimum confidence threshold for detections.
        classes=[PLAYER_CLASS_ID, BALL_CLASS_ID],  # Limits detection to players and the ball.
        verbose=False  # Disables detailed model logging.
    )  # Stores the model tracking results for this frame.

    ball_box = None  # Stores the detected ball bounding box if one is found.
    ball_track_id = -1  # Stores the ball tracking ID, or -1 if unavailable.
    player_candidates = []  # Stores detected player track IDs and boxes.

    # Process detections.
    if results[0].boxes is not None:  # Checks that the model returned at least one box collection.

        boxes = results[0].boxes  # Gets the detected/tracked boxes from the first result.

        for box in boxes:  # Iterates through each detected object box.

            # Get class ID.
            cls = int(box.cls[0])  # Converts the detected class ID to an integer.

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Converts the box corner coordinates to integers.
            track_id = int(box.id[0]) if box.id is not None else -1  # Reads the tracker ID, or -1 if missing.

            if cls == BALL_CLASS_ID:  # Handles detections identified as the ball.
                ball_box = (x1, y1, x2, y2)  # Saves the ball bounding box coordinates.
                ball_track_id = track_id  # Saves the ball tracker ID.

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draws a green box around the ball.
                draw_label(frame, f"Ball ID:{track_id}", x1, y1 - 10, (0, 255, 0))  # Labels the ball with its ID.

            elif cls == PLAYER_CLASS_ID:  # Handles detections identified as players.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)  # Draws an blue box around the player.
                draw_label(frame, f"Player ID:{track_id}", x1, y1 - 10, (255, 140, 0))  # Labels the player with its ID.
                player_candidates.append((track_id, box))  # Adds the player as a possession candidate.

    possession_text = "Possession: unknown"  # Sets the default possession display text.
    current_possession_player = None  # Stores the player currently judged to have possession.

    if ball_box is not None and player_candidates:  # Runs possession logic only when a ball and players exist.
        ball_center_x = (ball_box[0] + ball_box[2]) / 2.0  # Computes the ball center x-coordinate.
        ball_center_y = (ball_box[1] + ball_box[3]) / 2.0  # Computes the ball center y-coordinate.
        best_player = None  # Stores the closest player ID found so far.
        best_player_box = None  # Stores the closest player's bounding box.
        best_distance = float("inf")  # Starts with an infinite best distance.

        for track_id, player_box in player_candidates:  # Checks each player against the ball position.
            player_center_x, player_center_y = box_center(player_box)  # Computes the player's center point.
            distance = math.hypot(ball_center_x - player_center_x, ball_center_y - player_center_y)  # Measures ball-to-player distance.
            if distance < best_distance:  # Checks whether this player is closer than previous candidates.
                best_distance = distance  # Updates the shortest distance.
                best_player = track_id  # Updates the closest player ID.
                best_player_box = player_box  # Updates the closest player's box.

        player_scale = max(1.0, math.sqrt(box_area(best_player_box))) if best_player_box is not None else 1.0  # Estimates player size for distance scaling.
        max_allowed_distance = max(60.0, player_scale * POSSESSION_DISTANCE_FACTOR)  # Sets the possession distance limit.

        if best_player is not None and best_distance <= max_allowed_distance:  # Confirms possession if the closest player is near enough.
            current_possession_player = best_player  # Records the player currently in possession.
            possession_text = f"Possession: Player {best_player}"  # Updates the display text with the player ID.

            if best_player_box is not None:  # Checks that the possessing player's box is available.
                player_center_x, player_center_y = box_center(best_player_box)  # Computes the possessing player's center point.
                cv2.line(  # Starts drawing a line between the ball and possessing player.
                    frame,  # Provides the frame image to draw on.
                    (int(ball_center_x), int(ball_center_y)),  # Sets the line start at the ball center.
                    (int(player_center_x), int(player_center_y)),  # Sets the line end at the player center.
                    (0, 255, 255),  # Uses yellow for the possession line.
                    2,  # Sets the line thickness.
                )  # Finishes drawing the possession line.

    if current_possession_player is not None and last_possession_player is not None:  # Checks that current and previous possession both exist.
        if current_possession_player != last_possession_player:  # Detects a possession change between players.
            pass_count += 1  # Counts the possession change as a pass.

    if current_possession_player is not None:  # Checks whether possession was detected on this frame.
        last_possession_player = current_possession_player  # Stores this player for comparison on the next frame.

    draw_label(frame, possession_text, 20, 40, (0, 255, 255))  # Draws the possession status on the frame.
    draw_label(frame, f"Passes: {pass_count}", 20, 70, (0, 255, 255))  # Draws the pass count on the frame.
                
    cv2.imshow("Ball Tracking", frame)  # Shows the annotated frame in a window.
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Checks whether the user pressed the q key.
        print("Interrupted by user.")  # Prints a message when the user stops the program.
        break  
cap.release()  
cv2.destroyAllWindows()  
