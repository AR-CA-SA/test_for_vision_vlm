import os
import cv2
import base64
import requests
import signal
from openai import OpenAI
import time, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from formatter import to_json

PROMPT_1 = """

Here is an image of an object.

The object weight is: 216 g.

Based on the shape, volume, distribution and apparent materials in the image, please infer a single numerical value for the dominant frequency in Hz and amplitude (intensity) in dB of the noise the object will produce if it were to be dropped from height of 86cm onto hardwood.
 
For reference, the results should be relative to the measured ambient noise floor RMS = -59.23 dBFS. Treat this floor as the 0 dB baseline; report the impact's level above it

Take your best guess at an answer of the maximal expected noise. Your response MUST be in the following json format with no additional text:

{
DF : [your estimate of dominant frequency],
INDB : [your estimate of intensity],
"""






PROMPT_2 = """

Here is an image of an object.



The dimensions of the object are {Upper Diameter : 13.97 cm. Lower Diameter : 8.89 cm. Height : 0.75cm}.
And its weight is 216g.


Based on the shape, volume, distribution and apparent materials in the image, please infer a single numerical value for the dominant frequency in Hz and amplitude (intensity) in dB of the noise the object will produce if it were to be dropped from height of 86cm onto hardwood.
 
For reference, the results should be relative to the measured ambient noise floor RMS = -59.23 dBFS. Treat this floor as the 0 dB baseline; report the impact's level above it

Take your best guess at an answer of the maximal expected noise. Your response MUST be in the following json format with no additional text:

{
DF : [your estimate of dominant frequency],
INDB : [your estimate of intensity],
}

YOU MUST RESPOND WITH A SINGLE ESTIMATE FOR DOMINANT FREQUENCY AND INTENSITY in JSON format!


"""
# Signal handling for interruption
interrupted = False

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)


class OpenAIBase:
    def __init__(self):
        """
        Constructor method for initializing the `OpenAIBase` class.
        """
        ## Pull openai_key, create a client, and set the relative path
        self.key = os.environ.get("OPENAI_API_KEY")  # Fixed env var name
        if not self.key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.key)
        self.relative_path = 'Desktop/ComicCon'  # Updated for ComicCon

class VisionToText(OpenAIBase):
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self):
        '''
        Constructor method for initializing inherited class and a default image.
        '''
        super().__init__()

        ## Create directories if they don't exist
        self.setup_directories()

        ## Set default_image with proper error handling
        self.default_image = self._create_default_image()

    def setup_directories(self):
        """Create necessary directories"""
        base_path = os.path.join(os.environ['HOME'], self.relative_path)
        directories = ['images', 'prompts', 'temp']
        
        for directory in directories:
            dir_path = os.path.join(base_path, directory)
            os.makedirs(dir_path, exist_ok=True)
    
    def _create_default_image(self):
        """Create a simple default image if the original can't be loaded"""
        # Create a 300x300 gradient image as default
        default_img = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(300):
                default_img[i, j] = [
                    int(100 + (i/300) * 100),  # Red gradient
                    int(50 + (j/300) * 150),   # Green gradient  
                    200                         # Blue constant
                ]
        return default_img

    def viz_to_text(self, img='default', bbox=[0, 0, 640, 480], prompt_filename=None, prompt="what do you see?", max_length=1000):
        '''
        A function that performs vision-to-text conversion using OpenAI's API.
        Reference: https://platform.openai.com/docs/guides/vision

        Parameters:
        - img (Image or str): The image to analyze, either as an image object or a string for the default image.
        - prompt (str): The promp/home/prg/test_for_vision_vlmt/question to provide context for the image analysis.
        - bbox (list): The bounding box coordinates [x_min, y_min, x_max, y_max] to crop the image.
        - max_length (int): The maximum number of tokens for the response.
        '''
        ## Use the default image if 'img' is provided as a string
        if isinstance(img, str):
            img = self.default_image
        
        ## Use conditional statement to pull text from the prompt directory
        if prompt_filename != None:
            prompt_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', prompt_filename)
            try:
                with open(prompt_dir, 'r') as file:
                    prompt = file.read()
            except FileNotFoundError:
                print(f"Warning: Prompt file not found at {prompt_dir}, using default prompt")
    

        ## Crop the image using the provided bounding box coordinates
        cropped_image = img
        
        ## Define the temporary image file name, path, and save the cropped image
        img_name = 'temp.jpeg'
        temp_directory = os.path.join(os.environ['HOME'], self.relative_path, 'temp', img_name)
        
        # Convert RGB to BGR for cv2.imwrite if needed
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        else:
            cropped_image_bgr = cropped_image
            
        cv2.imwrite(temp_directory, cropped_image_bgr)

        ## Open the saved image file and encode it in base64 format
        with open(temp_directory, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            ## Set up the headers for the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ]
                    }
                ],
                "max_tokens": max_length
            }

            start = time.time()
            ## Send the POST request to OpenAI's API and retrieve the response and extract the content (text)
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                end = time.time()
                print(f"Received response after {round(end-start, 2)} seconds")
                data = response.json()
                if 'choices' in data.keys():
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = f"Error: {data}"
            except requests.exceptions.RequestException as e:
                content = f"API request failed: {e}"
            
        ## Remove the temporary image file
        try:
            os.remove(temp_directory)
        except FileNotFoundError:
            pass

        ## Return the extracted content
        return content


    



if __name__ == "__main__":
    
    vtt = VisionToText()

    object_name = "plate_216"

    resultsNoDimensions = []
    resultsWDimensions = []
    # resultsWReference = []


    img = cv2.imread("ceramicplate.jpg")
    
    


    for i in range(3):

        #ignore what is below this line but inside this loop for now



        resultsWDimensions.append(vtt.viz_to_text(img=img, prompt = PROMPT_2))
        # resultsWReference.append(vtt.viz_to_text(img=img, prompt = PROMPT_3))
    # to_json(resultsNoDimensions, "recorded_outputs_w_dimensions/w_dimensions_{object_name}".format(object_name=object_name))
    to_json(resultsWDimensions, "recorded_outputs_w_dimensions/w_dimensions_{object_name}".format(object_name=object_name))