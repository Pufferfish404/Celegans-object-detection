from ultralytics import YOLO

model = YOLO('yolov8n.pt')

#source can be a filepath or live camera (1)
results = model(source=1, show=True, save=True)
