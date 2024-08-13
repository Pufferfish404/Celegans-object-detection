from ultralytics import YOLO

model = YOLO(r'C:\Users\wangm3\Documents\Celegans-object-detection\runs\detect\train\weights\best.pt')

results = model.train(data=r"C:\Users\wangm3\Documents\Celegans-object-detection\Cards-3\data.yaml", epochs=20, imgsz=640, device="mps")