import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import logging
import time
import os

# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO('yolov8x.pt')

# Input Video
video_path = r"C:\Users\dhanu\Downloads\demopro\1338598-hd_1920_1080_30fps.mp4"
logger.info("Starting the video..")
cap = cv2.VideoCapture(video_path)

# Load class names
basedir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(basedir, "coco.txt")

with open(file_path, "r") as my_file:
    class_list = my_file.read().split("\n")


def get_person_coordinates(frame):
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a).astype("float")

    list_corr = []
    for _, row in px.iterrows():
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        d = int(row[5])
        c = class_list[d]
        if c == "person":
            list_corr.append([x1, y1, x2, y2])
    return list_corr


def people_counter():
    tracker = DeepSort(max_age=30)
    trackableObjects = {}
    unique_ids = set()

    totalFrames = 0
    totalUp = 0
    totalDown = 0
    move_in = []
    move_out = []

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("Final_output.mp4", fourcc, 30, (W, H))

    fps = FPS().start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if totalFrames % 3 != 0:
            totalFrames += 1
            continue

        frame = cv2.resize(frame, (500, 280))

        results = model.predict(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (0, H // 2 - 10), (W, H // 2 - 10), (0, 0, 0), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            unique_ids.add(track_id)

            l, t, r, b = map(int, track.to_ltrb())
            centroid = ((l + r) // 2, (t + b) // 2)

            to = trackableObjects.get(track_id)

            if to is None:
                to = TrackableObject(track_id, centroid)
            else:
                y_positions = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y_positions)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2 - 20:
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2 + 20:
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True

            trackableObjects[track_id] = to

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (255, 255, 255), -1)

        cv2.putText(frame, f"Enter: {totalDown}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"Exit: {totalUp}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        current_inside = len(move_in) - len(move_out)
        cv2.putText(frame, f"Inside: {current_inside}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(frame)
        cv2.imshow("People Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        totalFrames += 1
        fps.update()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    fps.stop()

    logger.info(f"Elapsed time: {fps.elapsed():.2f}")
    logger.info(f"FPS: {fps.fps():.2f}")


if __name__ == "__main__":
    people_counter()
