from ultralytics import YOLO
import cv2


from sort import *
from SubModule import get_car, read_license_plate


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('../YOLO_Weights/yolov8n.pt')
license_plate_detector = YOLO('../YOLO_Weights/model_ploy.pt')

# load video
cap = cv2.VideoCapture('../parking_video_footage/VID_20240304.mp4')


Yolo_ClassName = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
Car_Class_ID = [2, 7]
Bike_Class_ID = [3]
print(coco_model.info())

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_4wheeler = []
        detections_2wheeler = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in Car_Class_ID and score >0.9:
                detections_4wheeler.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            if int(class_id) in Bike_Class_ID and score >0.5:
                detections_2wheeler.append([x1, y1, x2, y2, score])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (105, 150, 0), 3)


        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_4wheeler))
        for track_id in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track_id
            cv2.putText(frame, f" {car_id}",
                    (max(0, int(xcar1)), max(30, int(ycar1))), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1 and score > 0.04:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                #cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 0), 3)
                #cv2.putText(frame, f" {car_id}",(max(0, int(xcar1)), max(30, int(ycar1))), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0, 0), 2)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("licence",license_plate_crop)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
                print(f"{license_plate_text} : {license_plate_text_score}")
                cv2.putText(frame,f"{license_plate_text} : {license_plate_text_score}",(max(0,int(x1)),max(30,int(y1))),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}



        cv2.imshow("Video", frame)
        cv2.waitKey(1)

# write results
#write_csv(results, './test.csv')
