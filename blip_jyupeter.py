import os
from PIL import Image
import torch
import pandas as pd
from IPython.core.display import display, HTML
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)


image_folder = "beautiful_person"  


image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
import os
from PIL import Image
import torch
import pandas as pd
from IPython.core.display import display, HTML
from transformers import Blip2Processor, Blip2ForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)


image_folder = "beautiful_person"  


image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


results = []


for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

   
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    img_tag = f'<img src="{image_path}" width="100"/>'
    results.append({"Image": img_tag, "Image Name": image_file, "Caption": caption})


df = pd.DataFrame(results)


html = df.to_html(escape=False)
display(HTML(html))  

results = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

   
    img_tag = f'<img src="{image_path}" width="100"/>'
    results.append({"Image": img_tag, "Image Name": image_file, "Caption": caption})


df = pd.DataFrame(results)


html = df.to_html(escape=False)
display(HTML(html))  
