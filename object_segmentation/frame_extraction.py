import cv2
import os
import time
import numpy as np
from argparse import ArgumentParser
from ultralytics.models.sam  import SAM3SemanticPredictor
overrides = dict(conf=0.25, task="segment", mode="predict" , model="./models/sam3.pt" ,  save=True)

DETECT_INTERVAL = 5
HOUSEHOLD_PROMPTS = [
    "cup", "mug", "plate" , "spoon" , "ceramic plate"
]

model_sam_3 = SAM3SemanticPredictor(overrides=overrides)
def video_frame_segmentation(pathIn,pathOut):
    count = 0
    video= cv2.VideoCapture(pathIn) 
    detect = 0
    now = time.time()
    while video.isOpened():
        suc, frame = video.read()
        if not suc:
            break

        cv2.imshow("current video" , frame)
         
        if now - detect >=  5:
            count +=1
            detect = now
            results = model_sam_3(source=pathIn, text=HOUSEHOLD_PROMPTS)

            if results and results[0].boxes is not None:
                out_path = os.path.join(pathOut, f"frame{count}.jpg")
                cv2.imwrite(out_path, frame )
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pathIn" ,required=True ,help="path to video")
    parser.add_argument("--pathOut" , required=True, help = "path to images")
    args = parser.parse_args()
    video_frame_segmentation(args.pathIn, args.pathOut)