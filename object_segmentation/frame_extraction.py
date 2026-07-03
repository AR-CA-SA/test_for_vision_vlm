import signal
import numpy as np
from colors import bcolors
import cv2
import os
import time
import torch
from argparse import ArgumentParser
import subprocess
import threading
from ultralytics.models.sam  import SAM3SemanticPredictor
from multiprocessing import Queue
from multiprocessing.managers import BaseManager

overrides = dict(conf=0.25, task="segment", mode="predict" , model="./models/sam3.pt", imgsz = 560)

DETECT_INTERVAL = 2
HOUSEHOLD_PROMPTS = [
    "spoon", "pan" , "plate", "mug"
]
shared_queue = Queue()



class QueueManager(BaseManager):
    pass


model_sam_3 = SAM3SemanticPredictor(overrides=overrides)


def save_detected_objects(frame, results, pathOut):
    print(bcolors.WARNING + f"{results[0].boxes}" + bcolors.ENDC)
    boxes = results[0].boxes
    names = results[0].names
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        crop = frame[y1:y2,x1:x2]
        cls_id = int(box[0].cls)
        if isinstance(names, dict):
            label = names.get(cls_id,str(cls_id)   )
        else:
            label = names[cls_id]
        if crop.size == 0 or crop.shape[0] < 100 or crop.shape[1] < 100:
            continue

        out_path = os.path.join(pathOut, f"object_{label}_number_{i}_size{crop.shape}.jpg")
        shared_queue.put(out_path)
        ok = cv2.imwrite(out_path, crop)


        if not ok:
            print(bcolors.FAIL + f"failed to write image_crop{i}_size{crop.shape}.jpg" + bcolors.ENDC )
        
def crop_image(frame):
    x , y , rgb = frame.shape
    print("before transformation : ", frame.shape)
    x_mid = x //2
    y_mid = y // 2
    x_start = x_mid - 200 
    x_end = x_mid + 200
    y_start = y_mid - 200
    y_end = y_mid + 200
    cropped_frame = frame[x_start:x_end, y_start:y_end]
    print("after transformation : " , cropped_frame.shape)
    return cropped_frame
def video_frame_segmentation(pathIn,pathOut):
    count = 0
    video= cv2.VideoCapture(pathIn) 
    detect = 0
    try:
        while video.isOpened():
            suc, frame = video.read()
            if not suc:
                break
            now = time.time()
            if now - detect >=  2:
                count +=1
                detect = now
                resized_frame = cv2.resize(frame, (1200,800), interpolation=cv2.INTER_AREA)
                frame_transform = crop_image(resized_frame)
                results = model_sam_3(source=frame_transform, text=HOUSEHOLD_PROMPTS)
                if results and results[0].boxes is not None:
                    save_detected_objects(frame_transform, results, pathOut)
    except KeyboardInterrupt:
        print("\n program terminated")       
    finally:
        video.release()
        cv2.destroyAllWindows()




def initialize_server():
    QueueManager.register('get_queue', callable=lambda: shared_queue)
    manager = QueueManager(address=('', 50000), authkey=b'abc')
    server = manager.get_server()
    server.serve_forever()   
def video(pathIn):
        video= cv2.VideoCapture(pathIn) 
        while video.isOpened():
            successs, frame = video.read()
            if not successs:
                print("error reading the video")
                break
            resized_frame = cv2.resize(frame, (1200,800), interpolation=cv2.INTER_AREA)
            cv2.imshow("video" , resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":



    print(torch.cuda.is_available())  
    print(torch.cuda.get_device_name(0))    
    parser = ArgumentParser()
    parser.add_argument("--pathIn" ,required=True ,help="path to video")
    parser.add_argument("--pathOut" , required=True, help = "path to images")
    args = parser.parse_args()
    t1 = threading.Thread(target=video,args=(args.pathIn,))
    t2 = threading.Thread(target=video_frame_segmentation,args=(args.pathIn,args.pathOut))
    t3 = threading.Thread(target=initialize_server)
    try:
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t33.join()
    finally:
        try:
           time.sleep(80800)
        except KeyboardInterrupt:
            print("\n program terminated")
            subprocess.run("rm -rf frames/*", shell= True ,  check = True)
            subprocess.run("rm -rf .runs/segment/*", shell= True ,  check = True)
  