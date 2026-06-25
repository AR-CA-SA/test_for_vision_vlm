import cv2 
import sys
import argparse
import numpy as np
import asyncio
import threading
import time
import sys
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

from ultralytics import YOLO


model = YOLO("yolo11n.pt")


shared_frame = SharedMemory(name="sharedFrame", size=8,create=True)



def main(video_url):
    image = np.array([])
    video = cv2.VideoCapture(video_url)
    success, frame = video.read()
    success, frame = video.read()
    cv2.im
    while success:
        video.set


if __name__ == "__main__":
    video_capture(video_url="test_video.mp4")
    shared_frame.close()
    shared_frame.unlink()