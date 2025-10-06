# app/processing/pipeline.py

import os
import cv2
import torch
from . import models
import uuid

from torchvision.transforms import transforms
from torchvision.utils import save_image
from .plate_detection import plate_detection_and_ocr  # ✅ import from same folder

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load trained deblurring model ONCE
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load(
    'app/models/model.pth',
    map_location=torch.device('cpu')
))

# Define preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def process_image(input_path):
    image = cv2.imread(input_path)
    orig_image = image.copy()
    orig_image = cv2.resize(orig_image, (224, 224))

    # Save blurred input
    output_dir = "app/results"
    os.makedirs(output_dir, exist_ok=True)

    original_out = os.path.join(output_dir, f"original_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(original_out, orig_image)

    # Deblur + process
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        deblurred_out = os.path.join(output_dir, f"deblurred_{uuid.uuid4().hex}.jpg")
        save_decoded_image(outputs.cpu().data, deblurred_out)

    # Now YOLO + OCR
    extracted_text, final_image_path = plate_detection_and_ocr(deblurred_out)
    return final_image_path, extracted_text

# output_path, text = process_image(r"testing_images\plate3.jpg")