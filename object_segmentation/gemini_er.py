import signal
import time
import os

import multiprocessing
import cv2

from multiprocessing import Process, Queue, Event
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager

q_b = Queue()

class QueueManager(BaseManager):
    pass




def data_streame_handler():
    
    
# client = genai.Client()

# uploaded_file = client.files.upload(file = "object_segmentation/for_vlm.jpeg")

# interaction  = client.interactions.create(
#     model = "gemini-3.5-flash", 
#     input = [
#         {"type":"text", "text" : "Given the image of the household object, I want you t"} ,
#         {"type" : "image", "uri" : uploaded_file.uri, "mime_type" : uploaded_file.mime_type}])
# print(interaction.output_text)


if __name__ == "__main__":
    QueueManager.register('get_queue')
    manager = QueueManager(address=('', 50000), authkey=b'abc')
    manager.connect()
    remote_queue = manager.get_queue()
    print(remote_queue.get(time = 5))