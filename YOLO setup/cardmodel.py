from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(data=r"C:\Users\wangm3\Documents\Celegans-object-detection\Cards-3\data.yaml", epochs=20, imgsz=640, device="mps")