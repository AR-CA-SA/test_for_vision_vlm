from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import BaseManager
from google import genai

PROMPT = """Given the image of the following household object,
predict the dominant frequency and intensinity of the sound at the moment of impact
if the household object were to be dropped from a distance of 70cm to the floor.
Output format:
{DF : {your_prediction} , IN : {your_prediction}}"""

class QueueManager(BaseManager):
    pass
    
PATH = ""
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


        
if __name__=="__main__":
    initialize_vlm(path_to_object_image="pan01.jpg")

