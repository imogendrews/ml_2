import os
from PIL import Image
import torch
import pandas as pd
from IPython.core.display import display, HTML
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

# Directory containing images
image_folder = "beautiful_person"  # Change to your folder path

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
import os
from PIL import Image
import torch
import pandas as pd
from IPython.core.display import display, HTML
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

# Directory containing images
image_folder = "beautiful_person"  # Change to your folder path

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Store results
results = []

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    # Process and generate caption
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Store filename, caption, and image path
    img_tag = f'<img src="{image_path}" width="100"/>'
    results.append({"Image": img_tag, "Image Name": image_file, "Caption": caption})

# Convert to Pandas DataFrame
df = pd.DataFrame(results)

# Render DataFrame as HTML with images
html = df.to_html(escape=False)
display(HTML(html))  # Works in Jupyter Notebook
# Store results
results = []

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    # Process and generate caption
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Store filename, caption, and image path
    img_tag = f'<img src="{image_path}" width="100"/>'
    results.append({"Image": img_tag, "Image Name": image_file, "Caption": caption})

# Convert to Pandas DataFrame
df = pd.DataFrame(results)

# Render DataFrame as HTML with images
html = df.to_html(escape=False)
display(HTML(html))  # Works in Jupyter Notebook
