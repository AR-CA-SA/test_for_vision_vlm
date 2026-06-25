from ultralytics import YOLO


model = YOLO("yolo11n.pt")
shared_frame = SharedMemory(name="SharedFrame", create=False)

