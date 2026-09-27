# openDOME

**Hear it, then see it.** A low-cost node that detects non-emitting (e.g. fibre-optic) drones using sound and light only. A 4-microphone array listens all the time; when it hears drone-band sound it wakes a webcam on a pan-tilt mount, and YOLO finds the drone and steers the servos to keep it centred.

Built for SDTH 2026, Challenge 02 (Creative Sensing). Full brief: [`creative-sensing-project-brief.md`](creative-sensing-project-brief.md).

> **Change of plan:** acoustic direction-finding (GCC-PHAT bearing, array calibration) has been **dropped** — we couldn't get reliable bearings. The mics are now a **wake-up trigger only**: they decide *when* to look, and YOLO decides *where*.

---

## How it works

```
ASLEEP  (camera off, YOLO idle)
  Mics ──► every 0.5 s: loudness of the 200–4000 Hz band (dBFS, loudest of the 4 mics)
     └─ score ≥ SCORE_THRESHOLD_DB for WAKE_CONSECUTIVE windows in a row
          → centre servos (P,90,90) → open camera → AWAKE

AWAKE   (mics still scored every 0.5 s)
  Camera ──► YOLO + ByteTrack ──► most confident drone ──► degrees off centre ──► P,pan,tilt steps
     └─ no loud window for 8 s AND no drone seen for 8 s → close camera → ASLEEP
```

- Sound alone wakes it; sound **or** a YOLO detection keeps it awake.
- Needing several loud windows in a row stops one-off claps and bangs from waking it.
- The YOLO model is loaded once at startup, so waking only costs the camera open time (~1 s).

---

## Status

| Part | State | File |
|---|---|---|
| YOLO drone detector (trained) | ✅ working | `train_yolo.py`, `best.pt` |
| Weather/occlusion augmentation for YOLO training | ✅ working | `augment_conditions.py` |
| Pan-tilt tracking from YOLO | ✅ working | `detect.py` |
| ESP32 firmware: 4-mic stream + servo commands | ✅ working | `pan_servo/src/main.cpp` |
| Mic level meter + 4-channel WAV recorder | ✅ working | `mic_monitor.py` |
| Mic wakes the camera (band-energy trigger) | ✅ working, threshold being tuned | `detect.py` |
| Drone-band loudness meter (for tuning the threshold) | ✅ working | `mic_level.py` |
| Acoustic direction-finding (GCC-PHAT, calibration) | ❌ dropped | — |
| Audio dataset | ⬜ to do (folder empty) | `dataset/sounds/` |
| Acoustic classifier (to replace the band-energy trigger) | ⬜ optional / later | `train_audio.py` (planned) |

---

## Hardware

- ESP32-S3 DevKitC-1 (N16R8)
- 4 × INMP441 I2S MEMS microphones on the corners of a 40 cm square, ESP32 in the centre (layout no longer matters now that bearing is dropped; the trigger uses whichever mic is loudest)
- 1080p USB webcam (mounted sideways, hence `ROTATE = "ccw"` in `detect.py`)
- 2 × MG995 180° servos (pan, tilt) on a pan-tilt bracket
- Separate 5–6 V servo supply with 1000 µF capacitors near the servos

### Array layout

```
   Mic 1 ●───────────● Mic 2
         │           │
         │   ESP32   │        40 cm square, mics facing up, foam covers on
         │           │
   Mic 4 ●───────────● Mic 3
         (camera faces up the page = 0°)
```

### Wiring

The firmware shares the I2S clock between both buses **inside the chip**, so no GPIO-to-GPIO jumper wires are needed.

| INMP441 pin | Mic 1 | Mic 2 | Mic 3 | Mic 4 |
|---|---|---|---|---|
| SCK | GPIO 4 | GPIO 4 | GPIO 4 | GPIO 4 |
| WS  | GPIO 5 | GPIO 5 | GPIO 5 | GPIO 5 |
| SD  | GPIO 6 | GPIO 6 | GPIO 7 | GPIO 7 |
| L/R | 3V3 | GND | 3V3 | GND |
| VDD | 3V3 | 3V3 | 3V3 | 3V3 |
| GND | GND | GND | GND | GND |

| Servo | Signal pin | Power |
|---|---|---|
| Pan  | GPIO 2 | separate 5–6 V supply |
| Tilt | GPIO 1 | separate 5–6 V supply |

- Mics run on **3.3 V only**. Put a 0.1 µF capacitor across VDD–GND at each mic.
- Each SD pair needs one mic with L/R→GND and one with L/R→3V3, or they talk over each other.
- Servo ground ties to ESP32 ground at **one point**. Never power servos from the ESP32 or laptop USB.

### Data link

Mic audio and servo commands share the ESP32-S3's **native USB** port (the one labelled USB, not UART).

- **ESP32 → laptop:** binary packets `A5 5A | seq u16 | rate u16 | frames u16 | frames×4×int16 | crc16`. 32 kHz, 256 frames per packet, 16-bit, samples interleaved `[line1 L, line1 R, line2 L, line2 R]`. `PacketReader` in `mic_monitor.py` parses them (and `detect.py` / `mic_level.py` import it from there).
- **Laptop → ESP32:** text lines `P,<pan>,<tilt>` in servo degrees, 90 = straight ahead / level.
- If the laptop falls behind (e.g. while YOLO runs a frame), the ESP32 drops packets rather than stalling; the `seq` gap shows how many. A few lost packets don't matter for the wake trigger. Lowering `SAMPLE_RATE` to 16000 in the firmware halves the traffic if it becomes a problem (the 200–4000 Hz band only needs 8 kHz+).

Only **one program** can open the port at a time. Close the PlatformIO serial monitor (and any other script) before running `detect.py`, `mic_level.py` or `mic_monitor.py`.

---

## Setup

```bash
pip install ultralytics opencv-python pyserial numpy pyyaml
```

Flash the firmware (PlatformIO, from `pan_servo/`):

```bash
cd pan_servo && pio run -t upload
```

Find the serial port (note the `*`) and set `SERIAL_PORT` in `detect.py`, `mic_level.py` and `mic_monitor.py`:

```bash
ls /dev/cu.usb*        # e.g. /dev/cu.usbmodem1301
```

The number can change if you use a different USB port. The PlatformIO upload log also shows it (`Uploading to /dev/cu.usbmodem…`).

---

## Running

| Command | What it does |
|---|---|
| `python open_external_webcam.py` | Camera preview only, to check the camera index (not tracked in git) |
| `python detect.py` | **Main program.** Sleeps listening to the mics; on drone-band sound wakes the camera, runs YOLO + ByteTrack and steers pan/tilt. Ctrl+C to quit (`q` also works while the window is open) |
| `python mic_level.py` | Prints the drone-band score every 0.5 s, scored exactly like `detect.py`. Use it to set the wake threshold |
| `python mic_monitor.py` | Live level meters for all 4 mics (clap test) |
| `python mic_monitor.py --record 60 --label park_s01` | Also save 60 s to `recordings/<time>_park_s01.wav` (4 channels = mics 1–4) |
| `python augment_conditions.py --preview 20` | Write 20 before/after examples of the weather/occlusion effects |
| `python augment_conditions.py` | Make degraded copies of 10% of `dataset/images/train` into `dataset/images/train_aug` |
| `python train_yolo.py` | Fine-tune YOLO on a mix of real + augmented images |

### Tuning `detect.py`

**Tracking:** camera wobbles → lower `GAIN`; camera lags → raise it (or `MAX_STEP`); jitters when centred → widen `DEADBAND`. If an axis runs away from the drone, flip `PAN_DIR` or `TILT_DIR`. `HFOV` is the field of view across the camera's long side.

**Wake-on-sound:**

| Setting | Current | Meaning |
|---|---|---|
| `SCORE_THRESHOLD_DB` | −41 | Band loudness (dBFS) that counts as a "loud window". Closer to 0 = louder |
| `WAKE_CONSECUTIVE` | 2 | Loud windows in a row needed to wake (2 = ~1 s of sustained sound) |
| `SLEEP_AFTER_QUIET_S` | 8.0 | Back to sleep after this long with no loud window **and** no detection |
| `BAND_LO`, `BAND_HI` | 200, 4000 Hz | Frequency band that is scored |
| `WINDOW_S` | 0.5 s | Length of each scored window |

If you change `BAND_LO`/`BAND_HI`/`WINDOW_S`, change them in `mic_level.py` too so the two agree.

---

## Setting the wake threshold

1. Close `detect.py`, run `python mic_level.py`.
2. Stay quiet for ~10 s, then fly the drone at the distance you care about. Ctrl+C.
3. Set `SCORE_THRESHOLD_DB` between the room level and the drone level, then raise `WAKE_CONSECUTIVE` if one-off noises still wake it.

**First test (indoors, 27 Sep):**

| Phase | band dB |
|---|---|
| Room before the drone | mostly −61 to −48, one spike to −39.5 |
| Drone flying, steady | −47 to −42 |
| Drone flying, loudest | up to −20 |

The margin between room and drone is only a few dB, so a far-away drone may not wake it. In the same test the **all-frequency** level jumped much more than the 200–4000 Hz band when the drone started (−41 → −29 dB, while the band only went −44 → −43 dB), which means most of the drone's sound is **outside** the current band — probably a high-pitched whine above 4 kHz. Next step: measure loudness per frequency range (<200, 200–1k, 1–4k, 4–8k, 8–16k Hz) quiet vs flying, and move `BAND_LO`/`BAND_HI` to the range that jumps most.

**Known effects to watch:**
- **Servo noise** may keep it awake forever (servos buzz right next to the mics). If so, only count loud windows while asleep: `if score_db >= SCORE_THRESHOLD_DB and not awake:` — staying awake then depends on YOLO alone.
- Talking, fans and aircon near the array are the likeliest false wakes. Test with them on.

---

## Mic bring-up

1. Run `mic_monitor.py`, clap next to each mic in turn; the loudest bar must match its label. Fix `MIC_SLOTS` in `mic_monitor.py` if not.
2. `mic_monitor.py --record 30`. Check `lost` and `crc` stay at 0, and all four tracks look right in Audacity. If packets drop, lower `SAMPLE_RATE` in the firmware.

---

## YOLO training

- Dataset: `dataset/images/` (`data.yaml`, one class `drone`). ~75k train, ~8.3k test images.
- `augment_conditions.py` makes degraded copies (rain, fog, sensor noise, motion blur, defocus, JPEG artefacts, low contrast, tree-canopy occlusion) with labels kept, into `dataset/images/train_aug/` (~7.5k images).
- `train_yolo.py` trains `yolo26n.pt` on a random 10% of the real images (`REAL_FRACTION`) plus all augmented ones, writing the list to `dataset/lists/`. It uses `device=0` (NVIDIA GPU); switch to `"mps"` on a Mac.
- `detect.py` loads `best.pt` from the repo root (copy the trained weights there).

---

## Acoustic dataset (optional, for a future classifier)

The band-energy trigger works now; a trained classifier would cut false wakes from fans, motorbikes etc. If we go there:

- Record **through the array** with `mic_monitor.py --record`, not a phone.
- **Drone:** 30–60 min of our own drone over several sessions, at several distances; supplement with public audio, e.g. [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset). If flying isn't permitted (check CAAS rules), play drone audio from a speaker.
- **Background:** several hours: park, expressway, aircraft approach, rain, insects, aircon, plus hard negatives (motorbikes, fans, lawnmowers, mosquitoes).
- Make each recording all drone or all background; label by folder and filename:

```
dataset/sounds/drone/2026-09-28_park_20m_hover_s01.wav
dataset/sounds/background/2026-09-28_expressway_s01.wav
dataset/sounds/sessions.csv    # file, date, location, weather, temp_c, label, drone_distance_m, notes
```

- Model plan: 1 s clips, log-mel spectrogram (64 bands), small CNN → P(drone). **Split by session/location, never by random clip.** Pick the threshold from false wakes per hour on held-out background, and keep the "N windows in a row" smoothing.

---

## What we report

| Metric | Owner |
|---|---|
| False wakes per hour, per environment (mic trigger alone) | Data & evaluation |
| Wake rate and wake distance for the drone at the chosen threshold | Acoustic lead |
| Vision confirmation accuracy; YOLO max range | Vision lead |
| Sound-to-lock latency: wake → camera open → first YOLO box (s) | Hardware / integration |
| First detected by: sound / vision, per environment | Data & evaluation |

---

## Next steps

1. Find the drone's frequency range (per-band `mic_level.py` run, quiet vs flying) and update `BAND_LO`/`BAND_HI` in `detect.py` and `mic_level.py`.
2. Re-tune `SCORE_THRESHOLD_DB` / `WAKE_CONSECUTIVE` outdoors, with realistic background noise and at real distances.
3. Check whether servo noise keeps it awake; apply the `and not awake` fix if so.
4. Measure false wakes per hour against recorded background noise.
5. Optional: record the audio dataset and train a classifier to replace the band-energy score.
