from ultralytics import YOLO
import cv2
import cvzone
import math

Yolo_ClassName = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

model = YOLO('../YOLO_Weights/yolov8n.pt')

def vehicledetection(img):
    check_licence_flag = False
    #success, img = Cam_obj.read()
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
                check_licence_flag = True

    return img, check_licence_flag

