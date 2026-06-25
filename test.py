from ultralytics import YOLO


model = YOLO("yolo12n.pt")

results = model.train(data="coco8.yaml", epochs=100,imgsz=640)