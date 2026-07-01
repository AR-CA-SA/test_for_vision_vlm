from multiprocessing.shared_memory import SharedMemory
import numpy as np
import json
from PIL import Image
from matplotlib import pyplot as plt

with open("frame_properties" , "r") as f:
    data = json.load(f)

dtype = data.get("dtype")
shape = tuple(data.get("shape")) 

existing_shared_mem = None
try:  
    
    existing_shared_mem =  SharedMemory(name = "shared_frame", create=False) 
    buffer = np.ndarray(shape, dtype = dtype, buffer=existing_shared_mem.buf)
    im = Image.fromarray(buffer)
    im.save("for_vlm.jpeg")
except KeyboardInterrupt:
     print("stop ... ")


