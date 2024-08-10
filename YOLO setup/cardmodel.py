from ultralytics import YOLO

model = YOLO('yolov8m.yaml')

results = model.train(data=r"C:\Users\wangs6\celegansobjdetect\Celegans-object-detection\Cards-3\data.yaml", epochs=20, imgsz=640, device="mps")