from collections import defaultdict

import cv2
import math
import time
import serial 
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/drone_yolo26n-3/weights/best.pt" # path to YOUR trained model
CAMERA = 0 # external webcam 
CONF = 0.3 # minimum confidence to count as a drone
IMGSZ = 640 # bigger = sees smaller drones but slower. Drop to 640 if it lags.
DEVICE = "mps" # Apple GPU. Use "cpu" if this errors.

HFOV = 100 # field of view across the camera's LONG side, in degrees
# If the webcam is mounted on its side (portrait), its picture arrives sideways.
# Look at the preview window: if the image is sideways, set this to "cw" or "ccw"
# (whichever makes it upright). Leave as None for a normal landscape camera.
ROTATE = "ccw" # camera turned 90 deg clockwise (as seen from the front), so turn the picture back anticlockwise
DEADBAND = 1 # only move if the drone is more than 2 deg off centre
GAIN = 0.4 # higher makes the pan faster, but more likely to overshoot. lower is smoother, but lag

# Which way the servos turn (from the p/t test):
#   p 120 turned the camera anticlockwise (LEFT), so a higher pan number = left  -> -1
#   t 170 tilted the camera UP, so a higher tilt number = up                     -> +1
# If the camera runs AWAY from the drone on one axis, flip that one's sign.
PAN_DIR = -1
TILT_DIR = 1
MAX_STEP = 3 # biggest move per command, in degrees: small steps = smooth, no blur
SEND_EVERY = 0.1 # seconds between servo commands, so the servo finishes moving before the next one
PAN_MIN, PAN_MAX = 0, 180    # narrow these if the camera or cable hits something
TILT_MIN, TILT_MAX = 0, 180
SERIAL_PORT = "/dev/cu.usbmodem1301" # plug in esp32, then run ls /dev/cu.usb*
ser = serial.Serial(SERIAL_PORT, 115200) # 115200 is the baud rate, must be same as esp32 code
pan = 90.0 # current pan angle, 90 = straight ahead
tilt = 90.0 # current tilt angle, 90 = level
ser.write(b"P,90,90\n") # start pointing straight ahead and level
last_send = 0.0 # time of the last servo command

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

trails = defaultdict(list)  # track id -> recent centre points

while True:
    ser.reset_input_buffer() # discard mic audio from the ESP32; we only send servo commands here
    ok, frame = cap.read()
    if not ok:
        print(f"Can't read from camera {CAMERA}")
        break
    if ROTATE == "cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE == "ccw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
            if c < CONF:
                continue

            # # convert the list of points into the format OpenCV
            # pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
            # cv2.polylines(annotated, [pts], False, (0, 255, 255), 2)
            x, y = int(x), int(y)
            diff_x, diff_y = x - cam_cx, y - cam_cy

            # remember the most confident drone; only that one steers the servo
            if best is None or c > best[1]:
                best = (diff_x, c, diff_y)

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
            f = (max(w, h) / 2) / math.tan(math.radians(HFOV / 2))   # focal length in pixels (same for x and y)
            ang_right = math.degrees(math.atan(best[0] / f))  # + = drone is right of centre
            ang_up = -math.degrees(math.atan(best[2] / f))    # + = drone is above centre (image y grows downward)
            moved = False
            if time.time() - last_send >= SEND_EVERY:
                if abs(ang_right) > DEADBAND:
                    step = max(-MAX_STEP, min(MAX_STEP, GAIN * ang_right))
                    pan = min(max(pan + PAN_DIR * step, PAN_MIN), PAN_MAX)
                    moved = True
                if abs(ang_up) > DEADBAND:
                    step = max(-MAX_STEP, min(MAX_STEP, GAIN * ang_up))
                    tilt = min(max(tilt + TILT_DIR * step, TILT_MIN), TILT_MAX)
                    moved = True
            if moved:
                last_send = time.time()
                ser.write(f"P,{pan:.1f},{tilt:.1f}\n".encode())   # e.g. "P,97.3,84.0\n"
                print(f"sent P,{pan:.1f},{tilt:.1f}  (drone {ang_right:+.1f} deg right, {ang_up:+.1f} deg up)")
                        
    cv2.imshow("Drone detection (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

ser.write(b"P,90,90\n") # back to straight ahead and level on exit
ser.close()
cap.release()
cv2.destroyAllWindows()
