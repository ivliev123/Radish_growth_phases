from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
data="yolo_dataset/data.yaml",
epochs=100,
imgsz=640,
batch=8,
device=0,
workers=0
)