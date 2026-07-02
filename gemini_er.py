import signal
import time
import os
from google import genai

client = genai.Client()

uploaded_file = client.files.upload(file = "object_segmentation/for_vlm.jpeg")

interaction  = client.interactions.create(
    model = "gemini-3.5-flash", 
    input = [
        {"type":"text", "text" : "Given the image of the household object, I want you t"} ,
        {"type" : "image", "uri" : uploaded_file.uri, "mime_type" : uploaded_file.mime_type}])
print(interaction.output_text)