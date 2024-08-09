from ultralytics import YOLO

model = YOLO('yolov8m.yaml')

results = model.train(data="/Users/RAX/Desktop/2024_cElegansObjectDetection/cloth_detection-5/data.yaml", epochs=20, imgsz=640, device="mps")