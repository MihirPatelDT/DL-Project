import string
import easyocr
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import re
from paddleocr import PaddleOCR

img_path = r"testing_images/plate11.jpg"

def plate_detection_and_ocr(img_path):
    model = YOLO(r"models\license_plate_detector.pt")
    results = model(img_path)

    # Load the original image with OpenCV
    img = cv2.imread(img_path)

    # YOLO returns results[0].boxes with x1, y1, x2, y2
    boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        
        # Crop license plate and OCR
        plate_img = img[y1:y2, x1:x2]
        result = ocr.predict(plate_img)
        print(result)
    
        if result:
            texts = []
            if not result[0]["rec_texts"]:
                texts.append("No text detected")
            else:
                for line in result:
                    for t in line["rec_texts"]:
                        texts.append(t)
            plate_number = " ".join(texts)
        
        print("Detected License Plate:", plate_number)
        
        # Overlay text on original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, plate_number, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()