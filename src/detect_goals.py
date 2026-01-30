import cv2
import json
from ultralytics import YOLO

VIDEO_PATH = "video/sample_10min.mp4"
OUTPUT_PATH = "output/goals.json"

# Manually define goal area (EDIT THIS)
GOAL_X1, GOAL_Y1 = 100, 200
GOAL_X2, GOAL_Y2 = 300, 400

BALL_CLASS_ID = 32  # YOLO class for sports ball
FRAMES_REQUIRED = 3  # how many frames ball must be inside goal

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
inside_counter = 0
goals = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, verbose=False)

    ball_found = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == BALL_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if GOAL_X1 <= cx <= GOAL_X2 and GOAL_Y1 <= cy <= GOAL_Y2:
                    inside_counter += 1
                    ball_found = True
                break

    if not ball_found:
        inside_counter = 0

    if inside_counter == FRAMES_REQUIRED:
        time_sec = frame_index / fps
        goals.append({
            "time_seconds": round(time_sec, 2)
        })
        inside_counter = 0

    frame_index += 1

cap.release()

with open(OUTPUT_PATH, "w") as f:
    json.dump(goals, f, indent=2)

print(f"Detected {len(goals)} goals")
