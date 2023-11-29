import cv2
import os
import xml.etree.ElementTree as ET
from datetime import datetime
def process_all_folders(base_folder):
    # Get a list of all folders in the base folder
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    print(subfolders)
    # Loop through each subfolder
    for subfolder in subfolders:
        input_folder = os.path.join(base_folder, subfolder, "mp4")
        output_folder = os.path.join(base_folder, subfolder, "jpg")
        print(input_folder)
        # Call the extract_frames_from_folder function for each subfolder
        extract_frames_from_folder(input_folder, output_folder)

def extract_frames_from_folder(input_folder, output_folder):
    # Get a list of all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # Loop through each video file in the folder
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        xml_file = os.path.join(input_folder, f"{os.path.splitext(video_file)[0]}.xml")

        # xml file에서 이상 행위가 발생한 시간대 검출 
        start_time, end_time = extract_time_from_xml(xml_file)
        print(start_time, end_time)
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue

        # Create a subfolder in the output folder for each video
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)

        # Get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Convert start_time and end_time to frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Read frames within the specified time range and save them as images
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            if frame_count>=start_frame and frame_count<=end_frame and frame_count%fps==0:
                # Save the frame as an image if it's within the frame interval
                frame_path = os.path.join(video_output_folder, f"[{os.path.splitext(video_file)[0]}]_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
            frame_count += 1

        # Release the video capture object
        cap.release()

        print(f"Frames extracted from {video_file}: {frame_count}")
        print(f"Frames saved in: {video_output_folder}")

def extract_time_from_xml(xml_file):
    # Parse the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    alpha=10

    # Extract start_time and end_time from the xml file (replace with your actual XML structure)
    start_time = root.find('.//StartTime').text
    start_time = convert_time(start_time)

    alarm_duration = root.find('.//AlarmDuration').text
    alarm_duration=convert_time(alarm_duration)
    end_time=start_time+alarm_duration
    return start_time-alpha, end_time

def convert_time(time_str):
    time_obj = datetime.strptime(time_str, "%H:%M:%S")

    # datetime 객체를 초 단위의 double 값으로 변환
    time_in_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    time_double = float(time_in_seconds)
    return time_double

# Example usage:
base_folder='data/valid'
process_all_folders(base_folder)
