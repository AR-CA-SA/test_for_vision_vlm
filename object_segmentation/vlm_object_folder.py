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
from PIL import Image
import pandas as pd
import glob
import re
import time 
import json
PATH_TO_IMAGES = "/home/prg/clip/object_images_single"
PATH_TO_MATERIAL = "/home/prg/clip/label_objects"
PROMPT_1 = """

The object shown in the image is {name} and the material is {material}.

Based on the shape, volume, mass distribution, and material of the object, infer a
single numerical estimate for:
1. The dominant frequency (Hz) of the sound the object would produce if struck at any
   point on its surface by an impact hammer (PCB 086C01). The hammer is designed to
   excite the object, not deform or break it.
2. The amplitude/intensity (dB) of that sound.

Intensity reference: 0 dB corresponds to the peak amplitude of the loudest impact
recorded anywhere across the full dataset. Typical values in this dataset range from
about -70 dB (very quiet objects) to about -30 dB (loud objects). Your intensity
estimate should be a negative integer.

Frequency reference: Typical values of frequency in the dataset range from 400 Hz to 14000 Hz 

Take your best guess at the maximal expected noise.

Respond with ONLY the following format, no additional text:
[<intensity_dB_int>, <frequency_Hz_int>]


"""


# Signal handling for interruption
interrupted = False

object_name_material = ["object_name", "material"]
db_f_cols = ["predicted_dB", "predicted_Hz"]

all_cols = object_name_material + db_f_cols
def get_vlm_predictions(image_paths, material_lookup, model):


    missing = []
    valid_paths = []

    rows =[]
    i = 0
    for img_path in image_paths:
        i +=1
        
  
        row = {}
        try:
            
            print(f"processing {img_path} ... ")
            img = cv2.imread(img_path)
 
            name_array = img_path.strip("/").split("/")
           
            material = material_lookup.get(name_array[-1])

            print(f"the material is {material}, and the object is {name_array[-1]}")
            
            predicted_value = model.viz_to_text(img=img, prompt = PROMPT_1.format_map({"name" : name_array[-1], "material" : material }))
            print(predicted_value)
            predicted_value_list = json.loads(predicted_value)

            row = {"object_name" : name_array[-1]}
            row.update({"material" : material})
            row.update({f"predicted_dB" : predicted_value_list[0]})
            row.update({f"predicted_Hz" : predicted_value_list[1]})

            rows.append(row)

            valid_paths.append(img_path)
            time.sleep(1)
        except Exception as e:

            for i in range(10):
                predicted_value = model.viz_to_text(img=img, prompt = PROMPT_1.format_map({"name" : name_array[-1], "material" : material }))
                if  isinstance(predicted_value, list):
                    time.sleep(2)
                    row = {"object_name" : name_array[-1]}
                    row.update({"material" : material})
                    row.update({f"predicted_dB" : predicted_value_list[0]})
                    row.update({f"predicted_Hz" : predicted_value_list[1]})
                    rows.append(row)
                    valid_paths.append(img_path) 
            missing.append(img_path)
            print(f"  BREAK LOOP, VLM MADE TOO MANY MISTAKES :  {img_path}, {material}: {e}")
            break
      
    df = pd.DataFrame(rows,columns= all_cols)
    if len(missing) > 0:
        with open("missing_paths.txt", "w") as txt_file:
            for path in missing:
                txt_file.write("".join(path) + "\n")

    
    return df




def build_material_lookup(folder_b_path):

    lookup = {}
    for material in os.listdir(folder_b_path):
        material_dir = os.path.join(folder_b_path, material)
        #check if the folder actually exists within folder_b_path
        if not os.path.isdir(material_dir):
            continue
        #map the name to the material
        for fname in os.listdir(material_dir):#list everything inside subfolder b
            full_path = os.path.join(material_dir, fname)
            if os.path.isfile(full_path):
                lookup[fname] = material.lower()
    return lookup


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


    


def extract_number(filepath):
    basename = os.path.basename(filepath)
    match = re.match(r'(\d+)', basename)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    
    vtt = VisionToText()

    object_name = "plate_average"

    resultsNoDimensions = []
    resultsWDimensions = []
    resultsWReference = []

    material_table = build_material_lookup(folder_b_path=PATH_TO_MATERIAL)
    print(material_table)


    print(f"\nLoading images from: {PATH_TO_IMAGES}")
    image_paths = glob.glob(f"{PATH_TO_IMAGES}/**/*.jpg", recursive=True)


    image_paths = sorted(image_paths, key=extract_number)






    print(image_paths)
    if not image_paths:
        image_paths = glob.glob(f"{PATH_TO_IMAGES}/*.jpg")
    
    if not image_paths:
        print("No images found! Check your path.")
        exit()
    

    img = cv2.imread("ceramicplate.jpg")
    
    df = get_vlm_predictions(image_paths=image_paths,material_lookup=material_table, model=vtt)

    df.to_csv("vlm_predicted_values.csv")