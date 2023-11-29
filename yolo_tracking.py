import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
import urllib.request


#VALID HIT+ URL+ CROP
def url_to_image(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype='uint8')
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  return image
def overlap(child_box,abuse_box):
    valid=1
    try:
        child_x,child_y,child_xw,child_yh=child_box
        abuse_x,abuse_y,abuse_xw,abuse_yh=abuse_box
        if(child_xw+child_x<abuse_x)|(child_yh+child_y<abuse_y)|(child_x>abuse_xw+abuse_x)|(child_y>abuse_yh+abuse_y):
            valid=0
    except:
        ValueError
    return valid

def crop_yolo(img, x,y,w,h,child_w,child_h,height,width):
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    blank_w = int((1.5)*(child_w))
    blank_h= int((1.5)*(child_h))
    ymin = y - blank_h
    ymax = y + h + blank_h
    xmin = x - blank_w
    xmax = x + w + blank_w
    if ymin < 0:
        ymin = 0
    if ymax > height:
        ymax = height
    if xmin < 0:
        xmin = 0
    if xmax > width:
        xmax = width
    crop = img[ymin:ymax, xmin:xmax]
    return crop,ymin,ymax,xmin,xmax
# 웹캠 신호 받기#
cap = cv2.VideoCapture('capstone_data/final/test/test (13).mp4')
#weights, cfg 파일 불러와서 yolo 네트워크와 연결
#net = cv2.dnn.readNet("capstone/weight/yolov4_final.weights", "capstone/cfg/yolov4-tiny_width.cfg")
net = cv2.dnn.readNet("capstone/weight/yolov4-tiny-augment-width.weights", "capstone/cfg/yolov4-tiny_width.cfg")


# YOLO NETWORK 재구성
classes = []
with open("capstone.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

idx=1
frame_cnt=0
while True:
    # 웹캠 프레임
    idx+=1
    child_w = 0
    child_h = 0
    frame_cnt+=1

    hit_w = 0
    hit_h = 0
    hit_x = 0
    hit_y = 0
    hit_flag = 0

    kick_x = 0
    kick_y = 0
    kick_h = 0
    kick_w = 0
    kick_flag = 0
    ret, frame = cap.read()
    #val = 30

    # 배열 더하기
    #frame = url_to_image('https://capstone-new.s3.ap-northeast-2.amazonaws.com/' + str(idx) + '.jpg')
    #array = np.full(frame.shape, (val, val, val), dtype=np.uint8)
    #frame = cv2.subtract(frame, array)
    #frame = cv2.resize(frame, (960, 720))
    frame = cv2.resize(frame, (16 * 80, 9 * 80))
    #height=9*80
    #width=16*80
    start = time.time()
    height, width, c = frame.shape
    #print(height)
    #print(width)
    child = []
    hit = []
    kick=[]
    child_flag = 0
    hit_flag = 0
    kick_flag=0
    valid = 1


    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0),
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
            #print(class_id)
            #print(confidence)
            # 검출 신뢰도
            if confidence > 0.1:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                dw = int(detection[2] * width)
                dh = int(detection[3] * height)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == ('child'):
                child = [x, y, w, h]
                score_child = confidences[i]
                child_flag = 1
                child_w = w
                child_h = h
                #print('child')
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)
            if label == ('hit'):
                hit = [x, y, w, h]
                score_hit = confidences[i]
                hit_flag = 1
                hit_w = w
                hit_h = h
                hit_x = x
                hit_y = y
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                print('hit: %.3f' %(score_hit))

            if label == ('kick'):
                kick = [x, y, w, h]
                score_kick = confidences[i]
                kick_flag = 1
                kick_w = w
                kick_h = h
                kick_x = x
                kick_y = y
                #print('kick: %3f' %(score_kick))
            end = time.time()
            t = end - start            # 경계상자와 클래스 정보 투영
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 150, 200), 5)
            #cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (200, 150, 200), 1)
    frame_cropped=frame.copy()
    if (hit_flag == 1 & child_flag == 1):
        valid = overlap(child, hit)
        try:
            cv2.rectangle(frame, (child[0], child[1]), (child[0] + child[2], child[1] + child[3]), (0, 255, 255), 5)  # 노란색
            cv2.rectangle(frame, (hit[0], hit[1]), (hit[0] + hit[2], hit[1] + hit[3]), (0, 255, 0), 5)  # 초록색
            frame_cropped,ymin,ymax,xmin,xmax = crop_yolo(frame, hit_x, hit_y, hit_w, hit_h, child_w, child_h, height, width)
            time.sleep(5)
            print(ymin, ymax, xmin, xmax)
            print('hit')
            print(frame_cnt)
        except:
            ValueError
    if (kick_flag == 1 & child_flag == 1):
        valid = overlap(child, kick)
        try:
            cv2.rectangle(frame, (child[0], child[1]), (child[0] + child[2], child[1] + child[3]), (0, 255, 255), 5)  # 노란색
            cv2.rectangle(frame, (kick[0], kick[1]), (kick[0] + kick[2], kick[1] + kick[3]), (0, 0, 255), 5)#빨간색
            frame_cropped,ymin,ymax,xmin,xmax = crop_yolo(frame, kick_x, kick_y, kick_w, kick_h, child_w, child_h, height, width)
            print('kick')
            print(frame_cnt)
        except:
            ValueError
    else:
        valid = 0
    #cv2.imshow('frame',frame)
    cv2.imshow('frame_cropped',frame_cropped)

    #cv2.imshow("YOLOv3", frame_cropped)
    # 1ms 마다 영상 갱신
    if cv2.waitKey(1) > 0:
        break
#out.release
