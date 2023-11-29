import cv2
import xml.etree.ElementTree as ET

# Read XML map file
tree = ET.parse('data/train/Loitering/mp4/C001201_004.xml')
root = tree.getroot() #최상위 경로 가져오기 

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
points_l=get_area('.//Loitering')
# Read video file
cap = cv2.VideoCapture('data/train/Loitering/mp4/C001201_004.mp4')

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    draw_area(frame,points,248,117,170)
    draw_area(frame,points_l,0,169,255)
    # Display the frame with edges
    cv2.imshow('Overlay', frame)

    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
