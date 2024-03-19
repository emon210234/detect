from ultralytics import YOLO

model = YOLO('best.pt')

results = model(source="example.jpg", show=True, conf=0.4, save=True)
