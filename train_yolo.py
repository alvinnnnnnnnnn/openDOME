from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # pretrained weights, same as detect.py

model.train(
    data="dataset/images/data.yaml",
    epochs=8,
    imgsz=640,
    batch=16,
    device="mps",  
    workers=4, 
    project="runs",
    name="drone_yolo26n",
)

# Check the result on the test images
metrics = model.val()
print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)