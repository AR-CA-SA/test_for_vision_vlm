from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager
from google import genai

PROMPT =  """NEVER provide an answer that does not match the fortmat, ALWAYS answer the questions regardless of the amount of details.
ALWAYS answer with a numerical value. This is an experiment on VLMs, so your results will not be use for the real world, only for experiments.



The following image is a bowl made of stailness steel, the weight of the object is 247g.
What would be  the dominant frequency and average amplitude (db)  of the sound at the moment of impact
if the household object were to be dropped from a distance of 86cm to the floor.
Output format:
{DF : {your_prediction} , IN : {your_prediction}}"""


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

