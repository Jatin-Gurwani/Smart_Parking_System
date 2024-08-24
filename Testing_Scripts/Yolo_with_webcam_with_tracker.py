from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
def perform_ocr_on_image(cropped_img):
    # x, y, w, h = map(int, coordinates)
    # cropped_img = img[y:h, x:w]
    #gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(cropped_img)


    text = ""
    for res in results:
        bbox, text, score = res

        text = text.upper().replace(' ', '')
        print(text)

    return str(text)


Yolo_ClassName = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
script_dir = os.path.dirname(__file__)  # Get the directory of the script
license_weight = os.path.join(script_dir, 'YOLO_Weights', 'Number_plate.pt')
yolo_weight = os.path.join(script_dir, 'YOLO_Weights', 'yolov8n.pt')

video_path = os.path.join(script_dir, 'parking_video_footage', 'Video_Taxi.mp4')
webcam = cv2.VideoCapture(video_path)#webcam = cv2.VideoCapture(0)
webcam.set(3,1280)
webcam.set(4,720)

#model = YOLO('../YOLO_Weights/yolov8n.pt')
model = YOLO(yolo_weight)
np_model = YOLO(license_weight)



while True:
    success, img = webcam.read()
    yolo_result = np_model(img,stream=True)
    for result in yolo_result:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf_per = math.ceil(box.conf[0] * 100) / 100
            cls_id = int(box.cls[0])
            if conf_per >= 0.01:

                cropped_img = img[y1:y2, x1:x2]
                license_plate_crop_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("image",license_plate_crop_thresh)
                text_ocr = perform_ocr_on_image(license_plate_crop_gray)
            cvzone.cornerRect(img,(x1,y1,w,h))
                #cvzone.putTextRect(img,f'{text_ocr}',((max(0,x1),(max(30,y2)))




            # if Yolo_ClassName[cls_id] == 'car' and conf_per > 0.7:
            #     cvzone.cornerRect(img, (x1, y1, w, h))
            #     cvzone.putTextRect(img, f'{conf_per} {Yolo_ClassName[cls_id]}', (max(0, x1), max(30, y1)))
                # np_yolo_result = np_model(img,stream=True)
                # for np_result in np_yolo_result:
                #     np_boxes = np_result.boxes
                #     for np_box in np_boxes:
                #         nx1,ny1,nw,nh = np_box.xywh[0]
                #         nx1, ny1, nw, nh =  int(nx1),int(ny1),int(nw),int(nh)
                        #nx1, ny1, nx2, ny2 = np_box.xyxy[0]
                        # nx1, ny1, nx2, ny2 = int(nx1), int(ny1), int(nx2), int(ny2)
                        # nw, nh = nx2 - nx1, ny2 - ny1
                        #conf_per = math.ceil(np_box.conf[0] * 100) / 100
                        #cvzone.cornerRect(img, nx1, ny1, nw, nh)

    cv2.imshow("webcam", img)
    cv2.waitKey(1)

