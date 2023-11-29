import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

#안 되는 경우 물체 하나만 정해서 직접 학습
#YOLO 네트워크 불러오기
def openpose(img, x,y,w,h,net_openpose):
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    blank = 30
    ymin = y - blank
    ymax = y + h + blank
    xmin = x - blank
    xmax = x + w + blank
    # print('%d %d %d %d'%(ymin, ymax, xmin, xmax))
    if ymin < 0:
        ymin = 0
    if ymax > height:
        ymax = height
    if xmin < 0:
        xmin = 0
    if xmax > width:
        xmax = width
    # print('[%d %d %d %d]'%(ymin, ymax, xmin, xmax))
    crop = img[ymin:ymax, xmin:xmax]
    imageHeight, imageWidth, _ = crop.shape
    # print(imageHeight)
    # print(imageWidth)
    # cv2.imshow('crop',crop)
    # cv2.waitKey(0)
    # network에 넣기위해 전처리
    # network 입력 blob 만들기
    inpBlob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    # network에 넣어주기
    # network 입력 설정
    net_openpose.setInput(inpBlob)
    layer_names_openpose = net_openpose.getLayerNames()
    output_layers_openpose = [layer_names[i - 1] for i in net_openpose.getUnconnectedOutLayers()]
    # 결과 받아오기
    # network 순방향 실행
    output_openpose= net_openpose.forward(output_layers_openpose)
    # 키포인트 검출시 이미지에 그려줌
    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비

    H = output_openpose.shape[2]
    W = output_openpose.shape[3]
    # print('H W : %d %d'%(H,W))
    points = []
    for i in range(0, 15):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output_openpose[0, i, :, :]
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x1 = (imageWidth * point[0]) / W
        y1 = (imageHeight * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            cv2.circle(crop, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1,
                       lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(crop, "{}".format(i), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            points.append((int(x1), int(y1)))  # point에 X,Y좌표 저장

        else:
            points.append(None)

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS_BODY_25:
        partA = pair[0]  # Head
        # print('partA: ',partA)
        partA = BODY_PARTS_BODY_25[partA]  # 0
        # print(partA)
        partB = pair[1]  # Neck
        # print('partB: ',partB)
        partB = BODY_PARTS_BODY_25[partB]  # 1
        # print(partB)
        # A-B

        if points[partA] and points[partB]:
            cv2.line(crop, points[partA], points[partB], (0, 255, 0), 2)
    return img

# 웹캠 신호 받기
VideoSignal = cv2.VideoCapture(0)

#print(img.shape)
#weights, cfg 파일 불러와서 yolo 네트워크와 연결
net_yolo = cv2.dnn.readNet("yolo/yolov3-tiny_best.weights", "yolo/yolov3-tiny.cfg")
# YOLO NETWORK 재구성
classes = []
BODY_PARTS_BODY_25 = {"Nose":0, "Neck":1,"RShoulder":2, "RElbow":3, "RWrist":4,
                      "LShoulder":5, "LElbow":6,  "LWrist":7,  "MidHip":8,  "RHip":9,
                      "RKnee":10,  "RAnkle":11, "LHip":12, "LKnee":13, "LAnkle":14,
                       "REye":15, "LEye":16,  "REar":17, "LEar":18, "LBigToe":19,
                      "LSmallToe":20, "LHeel":21, "RBigToe":22, "RSmallToe":23, "RHeel":24,  "Background":25}
POSE_PAIRS_BODY_25 = [["Nose", "Neck"], ["Nose", "REye"], ["Nose", "LEye"], ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["Neck", "MidHip"], ["MidHip", "RHip"], ["MidHip", "LHip"], ["RHip", "RKnee"], ["LHip", "LKnee"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["RKnee", "RAnkle"], ["LKnee", "LAnkle"], ["REye", "REar"], ["LEye", "LEar"], ["LAnkle", "LHeel"], ["LBigToe", "LHeel"], ["LSmallToe", "LHeel"],
                      ["RAnkle", "RHeel"], ["RBigToe","RHeel"], ["RSmallToe", "RHeel"]]

# 각 파일 path
# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = "openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = "openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"

with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#print(classes)
layer_names = net_yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]
flag=0
list_ball_location=[]
while True:
    # 웹캠 프레임
    ret, img = VideoSignal.read()
    height, width, c = img.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
    True, crop=False)
    net_yolo.setInput(blob)
    outs = net_yolo.forward(output_layers)


    # 위의 path에 있는 network 불러오기
    net_openpose = cv2.dnn.readNetFromCaffe(protoFile_body_25, weightsFile_body_25)

    img=cv2.GaussianBlur(img,(3,3),0,0)

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if(label=='person'):
                img=openpose(img,x,y,w,h,net_openpose)
            #score = confidences[i]
            # 경계상자와 클래스 정보 투영
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
            #cv2.putText(img, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            #cv2.imshow("YOLOv3", crop)
            #cv2.waitKey(0)
    cv2.imshow('image',img)
    if cv2.waitKey(1) > 0:
        break