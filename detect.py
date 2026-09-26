from collections import defaultdict

import cv2
import math
import serial 
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/drone_yolo26n-3/weights/best.pt" # path to YOUR trained model
CAMERA = 0 # external webcam 
CONF = 0.3 # minimum confidence to count as a drone
IMGSZ = 640 # bigger = sees smaller drones but slower. Drop to 640 if it lags.
DEVICE = "mps" # Apple GPU. Use "cpu" if this errors.

HFOV = 70
DEADBAND = 2 # only move if the drone is more than 2 deg off centre
GAIN = 0.4 # higher makes the pan faster, but more likely to overshoot. lower is smoother, but lag
SERIAL_PORT = "/dev/cu.usbmodem1301" # plug in esp32, then run ls /dev/cu.usb*
ser = serial.Serial(SERIAL_PORT, 115200) # 115200 is the baud rate, must be same as esp32 code
pan = 90.0 # current pan angle, 90 = straight ahead
ser.write(b"P,90\n") # start pointing straight ahead

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

trails = defaultdict(list)  # track id -> recent centre points

while True:
    ser.reset_input_buffer() # discard mic audio from the ESP32; we only send servo commands here
    ok, frame = cap.read()
    if not ok:
        print(f"Can't read from camera {CAMERA}")
        break

    # track() = detect + keep the same ID on the same drone from frame to frame
    results = model.track(
        frame, # image from the webcam, a numpy array in BGR format
        persist=True, # keep the tracker's memory between calls
        conf=CONF, # discards any detection below the confidence level 
        imgsz=IMGSZ, # size of the frame is resized from 1280x720 to 640x640 for the model, then the results are scaled back to 1280x720
        tracker="bytetrack.yaml", # picks the tracking algorithm
        device=DEVICE, # use either "cpu" or "mps" (Apple GPU). If you have an NVIDIA GPU, use "cuda"
        verbose=False # stops it printing a line of output for every frame.
    )
    # ultralytics multiple images at once. since only have one frame, use results[0]

    # frame is annotated with bounding boxes, class name, confidence and track ID
    # this is the frame that will be displayed 
    annotated = results[0].plot()  

    h, w = frame.shape[:2]
    cam_cx, cam_cy = w // 2, h // 2
    # Draw a cross at the centre of the camera
    cv2.drawMarker(annotated, (cam_cx, cam_cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # .boxes is the raw data behind each annotation
    # bounding box coordinates, confidence and track ID
    boxes = results[0].boxes
    if boxes.id is not None:
        best = None
        # Go through the drones one at a time. 
        # For each one, x, y is the centre of its box, w, h is the box's width and height, tid is its tracking ID.
        for (x, y, wb, hb), tid, c in zip(boxes.xywh.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
            # For each frame, append the current centre of the box to the list, and only keep past 30 pos to keep the trail short 
            # trails[tid].append((int(x), int(y)))
            # trails[tid] = trails[tid][-30:] 
            if c < 0.3:
                continue

            # # convert the list of points into the format OpenCV
            # pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
            # cv2.polylines(annotated, [pts], False, (0, 255, 255), 2)
            x, y = int(x), int(y)
            diff_x, diff_y = x - cam_cx, y - cam_cy

            # remember the most confident drone; only that one steers the servo
            if best is None or c > best[1]:
                best = (diff_x, c)

            # Draw a line from the centre of the camera to the centre of the bounding box
            cv2.line(annotated, (cam_cx, cam_cy), (x, y), (0, 0, 255), 2)
            cv2.putText(
                annotated, 
                f"ID {tid}: diff_x={diff_x} diff_y={diff_y}", 
                (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 255), 2
            )
        if best is not None:
            f = (w / 2) / math.tan(math.radians(HFOV / 2))   # focal length in pixels
            angle = math.degrees(math.atan(best[0] / f))     # degrees off centre (+ = right)
            if abs(angle) > DEADBAND:
                pan = min(max(pan + GAIN * angle, 0), 180)
                ser.write(f"P,{pan:.1f}\n".encode())         # e.g. "P,97.3\n"
                        
    cv2.imshow("Drone detection (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.write(f"P,{pan:.1f}\n".encode())
print(f"sent P,{pan:.1f}  (angle {angle:+.1f} deg)")
ser.close()
cap.release()
cv2.destroyAllWindows()
