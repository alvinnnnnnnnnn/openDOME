"""
Live drone-band loudness meter: use it to pick SCORE_THRESHOLD_DB for detect.py.

Prints one line every WINDOW_S seconds, scored exactly the way detect.py scores it.
    python mic_level.py
    python mic_level.py --port /dev/cu.usbmodem1301

1. Leave the room quiet for ~10 s  -> note the "band" numbers (room noise).
2. Fly the drone at the distance you care about -> note the "band" numbers again.
3. Set SCORE_THRESHOLD_DB in detect.py roughly halfway between the two.

Close detect.py / mic_monitor.py / the PlatformIO Serial Monitor first
(only one program can use the USB port at a time). Ctrl+C to stop.
"""
import argparse
import sys
import time

import numpy as np
import serial

from mic_monitor import PacketReader

SERIAL_PORT = "/dev/cu.usbmodem1301"   # find yours with: ls /dev/cu.usb*
WINDOW_S = 0.5                          # same as detect.py
BAND_LO, BAND_HI = 200, 4000            # same as detect.py


def band_score_db(samples, rate):
    """Loudness (dBFS) of the drone frequency band, loudest of the 4 mics. Same as detect.py."""
    x = samples.astype(np.float64)
    x -= x.mean(axis=0)
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    X[(freqs < BAND_LO) | (freqs > BAND_HI)] = 0
    y = np.fft.irfft(X, n=len(x), axis=0)
    rms = np.sqrt(np.mean(y ** 2, axis=0))
    return float(np.max(20 * np.log10(np.maximum(rms, 1e-3) / 32768)))


def full_db(samples):
    """Loudness (dBFS) of ALL frequencies, loudest mic - for comparison only."""
    x = samples.astype(np.float64)
    x -= x.mean(axis=0)
    rms = np.sqrt(np.mean(x ** 2, axis=0))
    return float(np.max(20 * np.log10(np.maximum(rms, 1e-3) / 32768)))


def bar(db, lo=-90, hi=-10, width=40):
    n = int(round((min(max(db, lo), hi) - lo) / (hi - lo) * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=SERIAL_PORT)
    args = ap.parse_args()

    ser = serial.Serial(args.port, 115200, timeout=0.05)
    reader = PacketReader()
    buf = []
    t0 = time.time()
    lowest = highest = None
    print("Listening... stay quiet first, then fly the drone. Ctrl+C to stop.\n")
    print("  time |  band dB |  all dB | band level (-90 .. -10 dB)")

    try:
        while True:
            for block in reader.feed(ser.read(ser.in_waiting or 1)):
                buf.append(block)
            if not reader.rate or sum(len(b) for b in buf) < reader.rate * WINDOW_S:
                continue
            samples = np.concatenate(buf)
            buf = []
            band = band_score_db(samples, reader.rate)
            lowest = band if lowest is None else min(lowest, band)
            highest = band if highest is None else max(highest, band)
            print(f"{time.time() - t0:6.1f} | {band:7.1f}  | {full_db(samples):6.1f}  | {bar(band)}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    if lowest is not None:
        print(f"\nQuietest window: {lowest:.1f} dB   loudest window: {highest:.1f} dB")
        print(f"Packets ok {reader.ok}, lost {reader.lost}, bad CRC {reader.crc_errors}")


if __name__ == "__main__":
    main()
