import cv2
import os
from ultralytics import YOLO

VIDEO_PATH = "../video/sport-union-1190-20260112-fussball-00.11.10.680-00.12.56.790.mkv"
print(os.path.exists(VIDEO_PATH))
BALL_CLASS_ID = 32  # sports ball

model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(VIDEO_PATH)

# total_frames = 0
# ball_detected_frames = 0
# consecutive = 0
# max_consecutive = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     total_frames += 1

#     results = model(frame, conf=0.25, verbose=False)

#     ball_found = False

#     for r in results:
#         for box in r.boxes:
#             if int(box.cls[0]) == BALL_CLASS_ID:
#                 ball_found = True
#                 break

#     if ball_found:
#         ball_detected_frames += 1
#         consecutive += 1
#         max_consecutive = max(max_consecutive, consecutive)
#     else:
#         consecutive = 0

# cap.release()

# print("=== BALL DETECTION REPORT ===")
# print(f"Total frames: {total_frames}")
# print(f"Frames with ball detected: {ball_detected_frames}")
# print(f"Detection rate: {ball_detected_frames / total_frames:.2%}")
# print(f"Max consecutive detections: {max_consecutive}")
