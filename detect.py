from collections import defaultdict

import cv2
import math
import sys
import time
import serial
import numpy as np
from ultralytics import YOLO

from mic_monitor import PacketReader  # safe to import: mic_monitor only runs main() when run directly

WEIGHTS = "best.pt" # path to YOUR trained model
CAMERA = 0 # external webcam
CONF = 0.6 # minimum confidence to count as a drone
IMGSZ = 640 # bigger = sees smaller drones but slower.
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

# --- wake-on-sound: the mic is only a trigger (no direction-finding) ---
WINDOW_S = 0.5             # score the mic every this many seconds
SCORE_THRESHOLD_DB = -40   # placeholder: tune it using the band score printed while asleep
WAKE_CONSECUTIVE = 2       # this many loud windows in a row are needed to wake (ignores one-off claps/bangs)
SLEEP_AFTER_QUIET_S = 8.0  # sleep again after this long with no sound AND no detection
BAND_LO, BAND_HI = 200, 4000   # Hz, where drone propeller noise mostly sits


def band_score_db(samples, rate):
    """Loudness (dBFS) of the drone frequency band, loudest of the 4 mics."""
    x = samples.astype(np.float64)
    x -= x.mean(axis=0)                               # remove DC offset
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    X[(freqs < BAND_LO) | (freqs > BAND_HI)] = 0      # keep only the drone band
    y = np.fft.irfft(X, n=len(x), axis=0)
    rms = np.sqrt(np.mean(y ** 2, axis=0))
    return float(np.max(20 * np.log10(np.maximum(rms, 1e-3) / 32768)))


def open_camera():
    c = cv2.VideoCapture(CAMERA)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return c


ser = serial.Serial(SERIAL_PORT, 115200) # 115200 is the baud rate, must be same as esp32 code
pan = 90.0 # current pan angle, 90 = straight ahead
tilt = 90.0 # current tilt angle, 90 = level
ser.write(b"P,90,90\n") # start pointing straight ahead and level
last_send = 0.0 # time of the last servo command

model = YOLO(WEIGHTS) # load once up front - it's slow, so do it before we need it

cap = None      # camera is only opened once the mic wakes us up
awake = False
mic_reader = PacketReader()
mic_buf = []
high_score_streak = 0
last_high_score_time = 0.0
last_detection_time = 0.0

trails = defaultdict(list)  # track id -> recent centre points

print("Idle - listening for drone-band sound (camera/YOLO off). Ctrl+C to quit.")

try:
    while True:
        # --- always: read the mic audio the ESP32 streams, and score it every WINDOW_S seconds ---
        for block in mic_reader.feed(ser.read(ser.in_waiting or 0)):
            mic_buf.append(block)
        if mic_reader.rate and sum(len(b) for b in mic_buf) >= mic_reader.rate * WINDOW_S:
            score_db = band_score_db(np.concatenate(mic_buf), mic_reader.rate)
            mic_buf = []
            now = time.time()
            if score_db >= SCORE_THRESHOLD_DB:
                high_score_streak += 1
                last_high_score_time = now
            else:
                high_score_streak = 0

            if not awake:
                sys.stdout.write(f"\rband score {score_db:5.1f} dB (threshold {SCORE_THRESHOLD_DB})   ")
                sys.stdout.flush()
                if high_score_streak >= WAKE_CONSECUTIVE:
                    print(f"\nDrone-band sound detected ({score_db:.1f} dB) - waking camera")
                    awake = True
                    pan, tilt = 90.0, 90.0
                    ser.write(b"P,90,90\n")    # look straight ahead; YOLO does the finding from here
                    cap = open_camera()
                    last_detection_time = now  # give YOLO a full SLEEP_AFTER_QUIET_S to find it

        if not awake:
            time.sleep(0.001)  # don't max out a CPU core while idle
            continue

        # --- awake: YOLO tracking + pan/tilt steering ---
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
                last_detection_time = time.time()  # drone in view keeps us awake even if it's quiet
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

        # --- back to sleep once it's been quiet AND nothing's been seen for a while ---
        now = time.time()
        if now - last_high_score_time > SLEEP_AFTER_QUIET_S and now - last_detection_time > SLEEP_AFTER_QUIET_S:
            print("Quiet and nothing seen - going back to sleep")
            awake = False
            high_score_streak = 0
            cap.release()
            cap = None
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # lets macOS actually close the window

except KeyboardInterrupt:
    pass
finally:
    ser.write(b"P,90,90\n") # back to straight ahead and level on exit
    ser.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print()
