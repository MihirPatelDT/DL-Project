import os
import cv2
import torch
import models
from torchvision.transforms import transforms
from torchvision.utils import save_image

# Ensure output directories exist
base_out_dir = "outputs/test_deblurred_images"
deblurred_dir = os.path.join(base_out_dir, "deblurred")
original_dir = os.path.join(base_out_dir, "original")

os.makedirs(deblurred_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load trained model
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load("outputs/model.pth", map_location=device))

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Save helper
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

# Path to test images
input_dir = "number_plates_image/blur_images"
all_images = sorted(os.listdir(input_dir))

print(f"Found {len(all_images)} images for testing.")

# Loop over all images
for idx, file_name in enumerate(all_images, start=1):
    # Read image
    img_path = os.path.join(input_dir, file_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"[WARN] Could not read {img_path}, skipping...")
        continue

    # Save original blurred (resized)
    orig_image = cv2.resize(image.copy(), (224, 224))
    orig_save_path = os.path.join(original_dir, f"original_{idx:04d}.jpg")
    cv2.imwrite(orig_save_path, orig_image)

    # Convert for model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(image)

    # Save deblurred
    deblur_save_path = os.path.join(deblurred_dir, f"deblurred_{idx:04d}.jpg")
    save_decoded_image(outputs.cpu().data, deblur_save_path)

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(all_images)} images")

print("✅ Testing complete. All deblurred and original images saved.")