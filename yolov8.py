import cv2
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# Read XML map file
tree = ET.parse('data/train/Falldown/mp4/C020100_004.xml')
root = tree.getroot() #최상위 경로 가져오기
# Load the YOLOv8 model
model = YOLO('model/pretrained/yolov8s.pt')


# Get detection area from XML
def get_area(xml_area):
    points = []
    for point in root.find(xml_area):# detect 'xml_area' -> iterate over the child elements
        x, y = map(int, point.text.split(',')) # get x,y coordinate by spliting comma & map: convert the split strings to integers
        points.append((x, y))
    return points

# Get detection area from XML
def draw_area(frame,points,R,G,B):
    # Draw edges on the frame
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (B,G,R), 5)
    cv2.line(frame, points[-1], points[0], (B,G,R), 5)
    return points

points=get_area('.//DetectArea')
#points_l=get_area('.//Intrusion')
# Read video file
cap = cv2.VideoCapture('openpose_test_video/input/violence.mp4')
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

size = (int(width), int(height))  # 프레임 크기

out = cv2.VideoWriter('openpose_test_video/output/violence_labeled.mp4', fourcc, fps,size)  # VideoWriter 객체 생성
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,classes=0)   #classes=0
        for result in results:
            boxes=result.boxes
       # draw_area(frame, points, 248, 117, 170)
        #draw_area(frame, points_l, 0, 169, 255)
        # Visualize the results on the frame
        frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
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
