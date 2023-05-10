# Import all necessary modules
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Loading pre-trained models
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setting up caption generation parameters
max_length = 16
num_beams = 4

# Defining function to generate captions for an image
def generate_captions(image_path, num_captions):
    try:
        # Opening and converting image to RGB format
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        
        # Extract image features and generate captions
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate specified number of captions
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_captions}
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # Update GUI with generated captions
        output_label.config(text="\n".join(preds))
    except Exception as e:
        output_label.config(text="Error: " + str(e))

# Define function to handle image selection
def select_image():
    # Open file dialog and get selected file path
    path = filedialog.askopenfilename()
    
    # Display selected image in GUI
    image = Image.open(path)
    image = image.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo
    
    # Generate caption(s) for selected image
    if num_captions_entry.get().isdigit():
        num_captions = int(num_captions_entry.get())
        if num_captions > 0:
            generate_captions(path, num_captions)
        else:
            output_label.config(text="Error: Number of captions must be greater than zero")
    else:
        generate_captions(path, 1)

# Set up GUI
root = tk.Tk()
root.title("Image Captioning Tool")

# Add number of captions input field
num_captions_label = tk.Label(root, text="Number of captions (optional):")
num_captions_label.pack(padx=10, pady=0)
num_captions_entry = tk.Entry(root)
num_captions_entry.pack(padx=10, pady=0)

# Add image upload button
upload_button = tk.Button(root, text="Upload Image", command=select_image)
upload_button.pack(padx=10, pady=10)


# Add image display label
image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)


# Add caption output label
output_label = tk.Label(root, text="Select an image to generate caption(s)")
output_label.pack(padx=10, pady=10)

root.geometry("800x600")
root.mainloop()
