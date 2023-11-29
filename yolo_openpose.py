import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO


#안 되는 경우 물체 하나만 정해서 직접 학습
#YOLO 네트워크 불러오기

def openpose(img, b):
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    #boxes:(xmin,ymin,xmax,ymax)
    box = b.tolist()
    xmin, ymin, xmax, ymax = map(int, box)
    crop = img[ymin:ymax, xmin:xmax]
    imageHeight, imageWidth, _ = crop.shape
    # network에 넣기위해 전처리
    # network 입력 blob 만들기
    inpBlob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    # network에 넣어주기
    # network 입력 설정
    net.setInput(inpBlob)
    # 결과 받아오기
    # network 순방향 실행
    output = net.forward()
    #print(output)
    # 키포인트 검출시 이미지에 그려줌
    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]
    # print('H W : %d %d'%(H,W))
    points = []
    for i in range(0, 25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x1 = (imageWidth * point[0]) / W
        y1 = (imageHeight * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            #cv2.circle(crop, (int(x1), int(y1)), 3, (0, 255, 255), thickness=-1,
                    #   lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            #cv2.putText(crop, "{}".format(i), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                      #  lineType=cv2.LINE_AA)
            points.append((int(x1), int(y1)))  # point에 X,Y좌표 저장

        else:
            points.append(None)

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        partA = pair[0]  # Head
        # print('partA: ',partA)
        partA = BODY_PARTS[partA]  # 0
        # print(partA)
        partB = pair[1]  # Neck
        # print('partB: ',partB)
        partB = BODY_PARTS[partB]  # 1
        # print(partB)
        # A-B

        if points[partA] and points[partB]:
            cv2.line(crop, points[partA], points[partB], (20,206,244), 2)
        cv2.imshow('crop', crop)
    return img
# Load the YOLOv8 model
model = YOLO('model/pretrained/yolov8s.pt')
names = model.names
'''BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_mpi = "openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_mpi = "openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"'''
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                   "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                   "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
                   "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
                   "RHeel": 24, "Background": 25 }
POSE_PAIRS = [["Neck", "Nose"], ["Neck", "RShoulder"],
              ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"],
              ["LEye", "LEar"], ["Neck", "MidHip"],
              ["MidHip", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["RAnkle", "RBigToe"],
              ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"],
              ["MidHip", "LHip"], ["LHip", "LKnee"],
              ["LKnee", "LAnkle"], ["LAnkle", "LBigToe"],
              ["LBigToe", "LSmallToe"], ["LAnkle", "LHeel"]]

protoFile = "openpose-master/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"
# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# webcam 사용시
cap = cv2.VideoCapture('openpose_test_video/input/violence.mp4')

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width), int(height))  # 프레임 크기
out = cv2.VideoWriter('openpose_test_video/output/violence_yolo_openpose.mp4', fourcc, fps,size)  # VideoWriter 객체 생성
# 웹캠 신호 받기
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,classes=0)   #classes=0

        for result in results:
            boxes=result.boxes
            for box in boxes:
                b=box.xyxy[0]
                c = names[int(box.cls)]
                if c=='person':
                    frame=openpose(frame,b)

        frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference",frame)
        out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            out.release()
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
