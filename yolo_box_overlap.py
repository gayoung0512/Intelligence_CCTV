import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

#안 되는 경우 물체 하나만 정해서 직접 학습
#YOLO 네트워크 불러오기
def overlap(child_box,hit_box):
    valid=1
    child_x,child_y,child_xw,child_yh=child_box
    hit_x,hit_y,hit_xw,hit_yh=hit_box
    if(child_xw<hit_x)|(child_yh<hit_y)|(child_x>hit_xw)|(child_y>hit_yh):
        valid=0
    return valid
# 웹캠 신호 받기
frame = cv2.imread('capstone_data/child_cctv/test/test.jpg')
h, w, c = frame.shape

#weights, cfg 파일 불러와서 yolo 네트워크와 연결
net = cv2.dnn.readNet("capstone/weight/yolov4-tiny_best-width.weights", "capstone/cfg/yolov4-tiny_width.cfg")
# YOLO NETWORK 재구성
classes = []
with open("capstone.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
flag=0
list_ball_location=[]

# YOLO 입력
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 검출 신뢰도
        if confidence > 0.5:
            # Object detected
            # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            dw = int(detection[2] * w)
            dh = int(detection[3] * h)
            # Rectangle coordinate
            x = int(center_x - dw / 2)
            y = int(center_y - dh / 2)
            boxes.append([x, y, dw, dh])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
child = []
hit = []
child_flag=0
hit_flag=0
valid=1
for i in range(len(boxes)):
    if i in indexes:
        label = str(classes[class_ids[i]])
        x, y, w, h= boxes[i]
        if label==('child'):
            child=[x, y, w, h]
            score_child = confidences[i]
            child_flag=1
        if label==('hit'):
            hit=[x, y, w, h]
            score_hit = confidences[i]
            hit_flag=1
        # 경계상자와 클래스 정보 투영
        cv2.circle(frame, (0,0), 3, (255, 255, 255), thickness=5,
                   lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        #cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
if(hit_flag==1 & child_flag==1):
    valid=overlap(child,hit)
else:
    valid=0
print(valid)
cv2.imshow("YOLOv3", frame)
# 1ms 마다 영상 갱신
cv2.waitKey(0)

