from ultralytics import YOLO
import cv2
import cvzone
import math

Yolo_ClassName = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']

webcam1 = cv2.VideoCapture(0)
webcam1.set(3, 429)
webcam1.set(4, 288)

webcam2 = cv2.VideoCapture(1)
webcam2.set(3, 640)
webcam2.set(4, 480)

import os

script_dir = os.path.dirname(__file__)  # Get the directory of the script
Background_image_path = os.path.join(script_dir, 'Images', 'background_image.png')
Background_image = cv2.imread(Background_image_path)
print('image :',Background_image)
#Background_image = cv2.resize(Background_image, (1280, 720))  # Adjust dimensions to match webcam frames


model = YOLO('../YOLO_Weights/yolov8n.pt')

while True:
    success1, img1 = webcam1.read()
    success2, img2 = webcam2.read()

    if success1 and success2:

        for img in [img1, img2]:
            yolo_result = model(img, stream=True)
            for result in yolo_result:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf_per = math.ceil(box.conf[0] * 100) / 100
                    cls_id = int(box.cls[0])
                    if Yolo_ClassName[cls_id] == 'cell phone':
                        cvzone.cornerRect(img, (x1, y1, w, h))
                        cvzone.putTextRect(img, f'{conf_per} {Yolo_ClassName[cls_id]}', (max(0, x1), max(30, y1)))

        Background_image[106:106+429, 240:240+288] = img1
        cv2.imshow("SmartParkTracker", Background_image)
        #cv2.imshow("webcam1", img1)
        cv2.imshow("webcam2", img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam1.release()
webcam2.release()
cv2.destroyAllWindows()
