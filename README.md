# AI image caption generator
AI tool that creates captions based on the image provided by the user.

This is a Python GUI application that generates captions for images using a pre-trained Vision Encoder-Decoder model from the Hugging Face Transformers library.

**Requirements**
This application requires the following modules to be installed:

tkinter
Pillow
torch
transformers
These modules can be installed via pip

**Usage**
To use the application, run the following command:
python image_captioning_tool.py

This will launch the GUI window, where you can select an image using the "Upload Image" button. The application will generate a single caption for the image by default, but you can choose to generate multiple captions by checking the "Generate Multiple Captions" checkbox.

Note that the application requires an internet connection to download the pre-trained model and tokenizer from the Hugging Face Model Hub.
