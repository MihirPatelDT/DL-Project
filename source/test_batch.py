import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import models
import torch
from Plate_Detection import plate_detection as pd

from torchvision.transforms import transforms
from torchvision.utils import save_image

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


# load the trained model
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load('models/model.pth',map_location=torch.device('cpu')))

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = r"testing_images/plate11.jpg"
image = cv2.imread(image_path)
orig_image = image.copy()
orig_image = cv2.resize(orig_image, (224, 224))
cv2.imwrite(f"output/original_blurred.jpg", orig_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image).unsqueeze(0)
print(image.shape)

with torch.no_grad():
    outputs = model(image)
    save_decoded_image(outputs.cpu().data, name=f"output/deblurred_image.jpg")
    pd.plate_detection_and_ocr(r'output/deblurred_image.jpg')


