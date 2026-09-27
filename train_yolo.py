# from ultralytics import YOLO

# import torch
# torch.cuda.empty_cache()

# def main(): 
#     model = YOLO("yolo26n.pt")  # pretrained weights, same as detect.py

#     model.train(
#         data="dataset/images/data.yaml",
#         epochs=25,
#         imgsz=640,
#         batch=8,
#         # device="mps",
#         device=0,  
#         workers=6, 
#         project="runs",
#         name="drone_yolo26n",
#         amp=False,      # compare speed against amp=True
#         val=False
#     )

#     # Check the result on the test images
#     metrics = model.val()
#     print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)                    

# if __name__ == "__main__":
#     main()

import random
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

DATA_YAML = Path("dataset/images/data.yaml")   # original: gives val set + class names
REAL_DIR = Path("dataset/images/train/images")
AUG_DIR = Path("dataset/images/train_aug/images")
REAL_FRACTION = 0.1                             # 30% of the real images, all augmented ones
SEED = 0                                        # same sample every run
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    return sorted(p.resolve() for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def resolve_val(cfg):
    """Turn the val entry of data.yaml into an absolute path."""
    val = Path(cfg["val"])
    if val.is_absolute():
        return val
    base = Path(cfg.get("path", ""))
    for root in (DATA_YAML.parent / base, Path.cwd() / base, DATA_YAML.parent):
        if (root / val).exists():
            return (root / val).resolve()
    raise FileNotFoundError(f"Can't find val folder {val} from {DATA_YAML}")


def build_mix_yaml():
    rng = random.Random(SEED)
    real = list_images(REAL_DIR)
    real = rng.sample(real, int(len(real) * REAL_FRACTION))
    aug = list_images(AUG_DIR)
    train = real + aug
    rng.shuffle(train)

    out_dir = Path("dataset/lists")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train_mix.txt"
    train_txt.write_text("\n".join(str(p) for p in train) + "\n")

    cfg = yaml.safe_load(DATA_YAML.read_text())
    cfg["val"] = str(resolve_val(cfg))
    cfg["train"] = str(train_txt.resolve())
    cfg.pop("path", None)
    cfg.pop("test", None)
    mix_yaml = out_dir / "data_mix.yaml"
    mix_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"Training on {len(real)} real + {len(aug)} augmented = {len(train)} images")
    return mix_yaml


def main():
    torch.cuda.empty_cache()
    mix_yaml = build_mix_yaml()

    model = YOLO("yolo26n.pt")  # pretrained weights, same as detect.py

    model.train(
        data=str(mix_yaml),
        epochs=25,
        imgsz=640,
        batch=8,
        # device="mps",
        device=0,
        workers=6,
        project="runs",
        name="drone_yolo26n_mix",   # separate folder from the non-augmented run
        amp=False,      # compare speed against amp=True
        val=False,
    )

    # Check the result on the full test images
    metrics = model.val()
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()