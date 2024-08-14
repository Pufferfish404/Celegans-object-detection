from ultralytics import YOLO

model = YOLO(r'C:\Users\wangm3\Documents\Celegans-object-detection\runs\detect\Aug14_2024_Cards\weights\best.pt')

#source can be a filepath or live camera (1)
results = model(source=r"C:\Users\wangm3\Documents\Celegans-object-detection\pokerhands.jpg", show=True, save=True)
