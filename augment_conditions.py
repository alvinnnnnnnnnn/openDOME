import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- effects
def rain(img, rng):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(h * w * rng.uniform(0.0008, 0.003))
    length = rng.randint(max(8, h // 60), max(12, h // 20))
    angle = np.deg2rad(rng.uniform(-20, 20))
    dx, dy = int(length * np.sin(angle)), int(length * np.cos(angle))
    xs = np.random.randint(0, w, n)
    ys = np.random.randint(0, h, n)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + dx), int(y + dy)),
                 float(rng.uniform(0.4, 1.0)), 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = img.astype(np.float32) * rng.uniform(0.75, 0.95)       # overcast
    out = out + layer[..., None] * rng.uniform(80, 160)
    return np.clip(out, 0, 255).astype(np.uint8)


def fog(img, rng):
    h, w = img.shape[:2]
    # Haze that is thicker at the top (distance/sky), with low-frequency variation.
    grad = np.linspace(1.0, rng.uniform(0.3, 0.8), h, dtype=np.float32)[:, None]
    blobs = cv2.resize(np.random.rand(8, 8).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    density = rng.uniform(0.3, 0.65) * grad * (0.7 + 0.3 * blobs)
    density = np.clip(density, 0, 0.85)[..., None]
    fog_color = rng.uniform(190, 235)
    out = img.astype(np.float32) * (1 - density) + fog_color * density
    return np.clip(out, 0, 255).astype(np.uint8)


def sensor_noise(img, rng):
    out = img.astype(np.float32)
    sigma = rng.uniform(5, 20)
    out += np.random.normal(0, sigma, out.shape)
    # Coloured chroma noise, like cheap cameras.
    if rng.random() < 0.5:
        chroma = np.random.normal(0, sigma * 0.5, (img.shape[0], img.shape[1], 1))
        out[..., rng.randrange(3)] += chroma[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, rng):
    k = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def defocus(img, rng):
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg(img, rng):
    q = rng.randint(12, 40)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def low_contrast(img, rng):
    alpha = rng.uniform(0.45, 0.75)
    mean = img.reshape(-1, 3).mean(0)
    out = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def canopy(img, rng, boxes):
    """Draw branches and leaf clusters, partly over each drone box.
    Covers at most ~45% of a box so the drone stays detectable."""
    h, w = img.shape[:2]
    out = img.copy()
    overlay = out.copy()
    mask = np.zeros((h, w), np.uint8)
    base = np.array([rng.uniform(20, 60), rng.uniform(35, 90), rng.uniform(20, 55)])

    def branch(x0, y0, x1, y1, thick):
        pts = np.linspace((x0, y0), (x1, y1), 6)
        pts[1:-1] += np.random.normal(0, max(2, thick * 1.5), (4, 2))
        cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thick, cv2.LINE_AA)

    def leaves(cx, cy, spread, size):
        for _ in range(rng.randint(4, 12)):
            x = int(cx + np.random.normal(0, spread))
            y = int(cy + np.random.normal(0, spread))
            ax = max(1, int(size * rng.uniform(0.5, 1.2)))
            ay = max(1, int(ax * rng.uniform(0.35, 0.6)))
            cv2.ellipse(mask, (x, y), (ax, ay), rng.uniform(0, 180), 0, 360, 255, -1)

    for cls, cx, cy, bw, bh in boxes:
        px, py, pw, ph = cx * w, cy * h, bw * w, bh * h
        if min(pw, ph) < 10:          # too small: occluding it would erase it
            continue
        box_mask = np.zeros_like(mask)
        box_mask[int(py - ph / 2):int(py + ph / 2), int(px - pw / 2):int(px + pw / 2)] = 1
        before = int((mask > 0).astype(np.uint8)[box_mask > 0].sum())
        # Branch crossing the box from outside, plus leaves at its end.
        side = rng.choice([-1, 1])
        x0, y0 = px + side * pw * rng.uniform(1.0, 3.0), py + ph * rng.uniform(-1.5, 1.5)
        x1, y1 = px + np.random.normal(0, pw * 0.25), py + np.random.normal(0, ph * 0.25)
        branch(x0, y0, x1, y1, max(1, int(min(pw, ph) * rng.uniform(0.05, 0.12))))
        leaves(x1, y1, min(pw, ph) * 0.3, min(pw, ph) * 0.18)
        cov = ((mask > 0).astype(np.uint8)[box_mask > 0].sum() - before) / box_mask.sum()
        if cov > 0.45:                # too much hidden: thin out the leaves
            erode = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
            mask = np.where(box_mask > 0, erode, mask)

    # A few extra branches elsewhere, so canopy isn't a "drone is here" cue.
    for _ in range(rng.randint(1, 4)):
        branch(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0, h),
               rng.randint(2, max(3, w // 150)))
        leaves(rng.uniform(0, w), rng.uniform(0, h), w * 0.03, w * 0.012)

    tex = np.random.normal(0, 12, (h, w, 3))
    overlay[:] = np.clip(base + tex, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255
    out = out * (1 - soft) + overlay * soft
    return np.clip(out, 0, 255).astype(np.uint8)


# Weighted menu of effects. Canopy is handled separately (it needs boxes).
EFFECTS = [
    (rain, 3),
    (fog, 3),
    (sensor_noise, 3),
    (motion_blur, 2),
    (defocus, 1),
    (jpeg, 2),
    (low_contrast, 2),
]


def apply_random(img, boxes, rng, canopy_p):
    names = []
    if boxes and rng.random() < canopy_p:
        img = canopy(img, rng, boxes)
        names.append("canopy")
    fns, weights = zip(*EFFECTS)
    k = rng.choice([1, 1, 2]) if names else rng.choice([1, 2, 2])
    chosen = []
    while len(chosen) < k:
        f = rng.choices(fns, weights)[0]
        if f not in chosen:
            chosen.append(f)
    # Compression / noise last, like a real camera pipeline.
    order = {rain: 0, fog: 0, low_contrast: 1, motion_blur: 2, defocus: 2,
             sensor_noise: 3, jpeg: 4}
    for f in sorted(chosen, key=order.get):
        img = f(img, rng)
        names.append(f.__name__)
    return img, names


# ---------------------------------------------------------------- dataset I/O
def read_boxes(label_path):
    """Return list of boxes, or None if the file has segment rows (skip those)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            return None
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def process(job):
    img_path, lbl_path, out_img_dir, out_lbl_dir, seed, canopy_p, preview = job
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 32))
    boxes = read_boxes(lbl_path)
    if boxes is None:
        return "skipped"
    img = cv2.imread(str(img_path))
    if img is None:
        return "skipped"
    aug, names = apply_random(img, boxes, rng, canopy_p)
    stem = f"{img_path.stem}_aug"
    if preview:
        h, w = img.shape[:2]
        panels = []
        for im in (img, aug):
            im = im.copy()
            for _, cx, cy, bw, bh in boxes:
                cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                              (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 0, 255), 2)
            panels.append(im)
        side = np.hstack(panels)
        cv2.putText(side, "+".join(names), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), side)
    else:
        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl_dir / f"{stem}.txt").write_text(
            lbl_path.read_text() if lbl_path.exists() else "")
    return "+".join(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="dataset/images/train",
                    help="folder containing images/ and labels/")
    ap.add_argument("--dst", default="dataset/images/train_aug")
    ap.add_argument("--fraction", type=float, default=0.1,
                    help="share of source images to make a degraded copy of")
    ap.add_argument("--canopy-p", type=float, default=0.3,
                    help="chance an image gets canopy occlusion")
    ap.add_argument("--preview", type=int, default=0,
                    help="write N before/after comparisons to <dst>_preview instead")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    imgs = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    rng = random.Random(args.seed)
    n = args.preview if args.preview else int(len(imgs) * args.fraction)
    picked = rng.sample(imgs, min(n, len(imgs)))

    if args.preview:
        out_img = Path(str(args.dst) + "_preview")
        out_lbl = out_img
    else:
        out_img, out_lbl = Path(args.dst) / "images", Path(args.dst) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    jobs = [(p, src / "labels" / f"{p.stem}.txt", out_img, out_lbl,
             args.seed * 1_000_003 + i, args.canopy_p, bool(args.preview))
            for i, p in enumerate(picked)]
    print(f"{len(jobs)} of {len(imgs)} images -> {out_img}")

    counts = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
            for name in res.split("+"):
                counts[name] = counts.get(name, 0) + 1
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    print("effect counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":   # required on Windows (same issue as your training script)
    main()
