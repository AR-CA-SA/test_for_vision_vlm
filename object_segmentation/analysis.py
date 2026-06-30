from ultralytics import YOLO
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import json

import cv2

model = YOLO("yolo26n.pt")


with open("frame_properties" , "r") as file:
    data = json.load(file)

print(data.get("size"))
size = list(data.get("size"))
shape = np.dtype(data.get("shape")) 

existing_shared_mem =  SharedMemory(name = "shared_frame", create=False)
buffer = np.ndarray(shape, dtype = size, buffer=existing_shared_mem.buf)

cv2.imshow(existing_shared_mem)


# cv2.imshow("",frame)
existing_shared_mem.close()
existing_shared_mem.unlink()