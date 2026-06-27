from ultralytics import YOLO
from multiprocessing.shared_memory import SharedMemory
import cv2

model = YOLO("yolo11n.pt")




existing_shared_mem =  SharedMemory(name = "sharedFrame", create=False)


# cv2.imshow("",frame)
shared_frame.close()
shared_frame.unlink()