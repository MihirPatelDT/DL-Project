# # import easyocr
# import matplotlib
# matplotlib.use("TkAgg")   # or "Agg" if no GUI

# import torch
# from matplotlib import pyplot as plt
# from ultralytics.nn.tasks import DetectionModel

# torch.serialization.add_safe_globals([DetectionModel])

# from ultralytics import YOLO
# model = YOLO(r"models\license_plate_detector.pt")
# results = model(r"outputs\test_deblurred_images\deblurred\deblurred_0008.jpg")  # detect license plate
# plt.imshow(results[0].plot())
# plt.axis("off")
# plt.show() # plot results

import easyocr
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import re

# Load YOLO model
model = YOLO(r"models\license_plate_detector.pt")

# Detect license plate
results = model(r"outputs\test_deblurred_images\deblurred\deblurred_6000.jpg")

# Load the original image with OpenCV
image_path = r"outputs\test_deblurred_images\deblurred\deblurred_6000.jpg"
img = cv2.imread(image_path)

# YOLO returns results[0].boxes with x1, y1, x2, y2
boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# for box in boxes:
#     x1, y1, x2, y2 = map(int, box)
    
#     # Crop the license plate
#     plate_img = img[y1:y2, x1:x2]
    
#     # Convert BGR to RGB for display with matplotlib
#     plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    
#     # Apply OCR
#     ocr_result = reader.readtext(plate_img_rgb)
#     print("OCR Result:", ocr_result)
    
#     # Display cropped license plate
#     plt.imshow(plate_img_rgb)
#     plt.axis('off')
#     plt.show()


for box in results[0].boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box)
    
    # Crop license plate
    plate_img = img[y1:y2, x1:x2]
    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    
    # OCR
    ocr_result = reader.readtext(plate_img_rgb)
    
    if ocr_result:
        ocr_result.sort(key=lambda x: x[2], reverse=True)
        raw_text = ocr_result[0][1]
        # Clean text: remove non-alphanumeric and uppercase
        plate_number = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    else:
        plate_number = "No text detected"
    
    print("Detected License Plate:", plate_number)
    
    # Overlay text on original image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, plate_number, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display final image with detection and OCR overlay
plt.figure(figsize=(12,8))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()



# import sys
# print(sys.executable)
