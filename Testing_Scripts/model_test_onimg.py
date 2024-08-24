import cv2
from ultralytics import YOLO
from SubModule import read_license_plate

# Constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)

# Load pre-trained YOLOv8 model
model = YOLO("../YOLO_Weights/Number_plate.pt")

# Load an image
image_path = "../Images/Bike_DL_08.jpg"  # Replace with your image file
image = cv2.imread(image_path)

# Perform object detection
results = model.predict(image, conf=CONFIDENCE_THRESHOLD, classes=0)

# Draw bounding boxes on detected objects
# for r in results:
#     for box in r.boxes:
#         b = box.xywh
#         cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 2)
for result in results:
    for routput in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = routput
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
        cv2.putText(image,f"{license_plate_text}  ,{license_plate_text_score}",(max(0,int(x1)),max(30,int(y1))),cv2.FONT_HERSHEY_SIMPLEX,1,GREEN,2)
# Save the output image
output_path = "output_image8.jpg"
cv2.imwrite(output_path, image)

print(f"Detected {len(results)} objects. Saved output to {output_path}")
