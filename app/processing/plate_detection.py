# app/processing/plate_detection.py

import os
import cv2
import uuid
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ✅ Load models only once
yolo_model = YOLO("app/models/license_plate_detector.pt")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_dir="app/models/PP-OCRv5_server_det",
    text_recognition_model_dir="app/models/PP-OCRv5_server_rec",
)

def plate_detection_and_ocr(img_path):
    results = yolo_model(img_path)

    # Load the original image with OpenCV
    img = cv2.imread(img_path)

    # YOLO returns results[0].boxes with x1, y1, x2, y2
    boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    plate_number = "No text detected"

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        plate_number = ""

        # Crop license plate and OCR
        plate_img = img[y1:y2, x1:x2]
        result = ocr.predict(plate_img)
    
        if result:
            texts = []
            for line in result:
                for t in line["rec_texts"]:
                    texts.append(t)
            plate_number = " ".join(texts)
        else:
            plate_number = "No text detected"
        
        print("Detected License Plate:", plate_number)
        
        # Overlay text on original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(img, plate_number, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    output_dir = "app/results"
    os.makedirs(output_dir, exist_ok=True)

    final_img_path = os.path.join(output_dir, f"detected_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(final_img_path, img)

    final_text = "".join(plate_number)
    return final_text, img_path
