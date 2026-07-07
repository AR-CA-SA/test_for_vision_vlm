from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager
from google import genai

PROMPT = """Imagine you are a personal robot that is trying to minimize the amount of noise that they make when interacting with objects. Using the image provided, identify the object and predict the potential Dominant Frequency and Intensity of the noise that 
the object would make if it were to fall from a latitude of height
 of 50cm towards a hard wooden floor, no dimenons are given but infer them based on the object"""

class QueueManager(BaseManager):
    pass
    

def initialize_vlm(path_to_object_image):


    client = genai.Client()

    uploaded_file = client.files.upload(file = path_to_object_image)
    interaction  = client.interactions.create(
        model = "gemini-3.5-flash", 
        input = [
            {"type":"text", "text" : f"{PROMPT}"} ,
            {"type" : "image", "uri" : uploaded_file.uri, "mime_type" : uploaded_file.mime_type}])
    print(interaction.output_text)
    return interaction.output_text

def data_stream_handler():
    QueueManager.register('get_queue')
    manager = QueueManager(address=('', 50000), authkey=b'abc')
    manager.connect()
    remote_queue = manager.get_queue()
    try:
        path_to_object_image = remote_queue.get(timeout = 2)
        output = initialize_vlm(path_to_object_image)
        print("Received path to object image from shared queue:", output)
    except Exception as e:
        print(f"is the queue empty? {remote_queue.empty()}  {e}")
        
if __name__=="__main__":
    data_stream_handler()

