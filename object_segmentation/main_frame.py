import cv2 
from argparse import ArgumentParser
import numpy as np
import threading
import time
import os
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import subprocess
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
shared_frame = SharedMemory(name="sharedFrame", size=8,create=True)
def video(pathIn):

    video = cv2.VideoCapture(pathIn)
    # org_frame = success, frame = video.read()
    # org_shape = np.shape(frame)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            print("died")
            break
        resized_frame = cv2.resize(frame, (1200,800), interpolation=cv2.INTER_AREA)
        results = model(resized_frame)

        cv2.imshow("Resized_Window", results[0].plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

def extract_frame(pathIn, pathOut):
    os.makedirs(pathOut, exist_ok=True)
    count = 0
    video = cv2.VideoCapture(pathIn)
    success = True
    while video.isOpened():
        video.set(cv2.CAP_PROP_POS_MSEC, (count*10*10**2))
        success , frame = video.read()
        if not success or frame is None or frame.size == 0:
            break
        frame_array = np.array(frame) 
        shared_np_mem = np.ndarray(frame_array.shape, frame_array.dtype, buffer=shared_frame.buf)
        shared_np_mem[:] = frame_array[:]
        out_path = os.path.join(pathOut, f"frame{count}.jpg")
        results = model(frame)
        cv2.imwrite(out_path, results[0].plot())
        count  += 5   
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pathIn" ,required=True ,help="path to video")
    parser.add_argument("--pathOut" , required=True, help = "path to images")
    args = parser.parse_args()
    t1 = threading.Thread(target=extract_frame, args = (args.pathIn, args.pathOut))
    t2 = threading.Thread(target=video, args= (args.pathIn,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    subprocess.run("rm -rf frames/*", shell= True ,  check = True)

    shared_frame.close()
    shared_frame.unlink()