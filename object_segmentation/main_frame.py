import cv2 
import numpy as np
import  asyncio
import threading
import time
import sys
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory


shared_frame = SharedMemory(name="sharedFrame", size=8,create=True)



def video_capture(video_url):
    cap = cv2.VideoCapture(video_url)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        cv2.imshow("s",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__name__":
    video_capture(video_url="test_video.mp4")
