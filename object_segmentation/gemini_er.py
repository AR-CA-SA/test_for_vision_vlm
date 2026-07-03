import signal
import time
import os

import multiprocessing
import cv2
import threading 
from multiprocessing import Process, Queue, Event
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager):
    pass

# def initialize_vlm(path_to_object_image):
#     client = genai.Client()
#     uploaded_file = client.files.upload(file = "object_segmentation/for_vlm.jpeg")
#     interaction  = client.interactions.create(
#         model = "gemini-3.5-flash", 
#         input = [
#             {"type":"text", "text" : "Given the image of the household object, I want you t"} ,
#             {"type" : "image", "uri" : uploaded_file.uri, "mime_type" : uploaded_file.mime_type}])
#     return interaction.output_text

def data_stream_handler():
    QueueManager.register('get_queue')
    manager = QueueManager(address=('', 50000), authkey=b'abc')
    manager.connect()
    remote_queue = manager.get_queue()
    path_to_object_image = remote_queue.get(timeout = 2)
    print("Received path to object image from shared queue:", path_to_object_image)

if __name__=="__main__":
    data_stream_handler()

