import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import tempfile

# Defining classes using a dictionary and list
class_map = {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Load the best weights from the trained model
trained_detection_model = YOLO('/path/to/best.pt')  # Update path as needed

# Streamlit app
st.title("PPE Detection with YOLOv8")

# Create a video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open video capture.")
else:
    stframe = st.empty()
    stop_button = st.button('Stop Stream')

    while True:
        if stop_button:
            st.write("Stream stopped by user.")
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("End of video or failed to capture frame.")
            break

        # Run YOLOv8 tracking on the frame
        results = trained_detection_model.track(frame, persist=True, conf=0.4, classes=[0,1,2,3,4,5,6,7])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        class_label_list = []

        for box in results[0].boxes:  # Iterate over bounding boxes in the frame
            class_id = int(box.cls)  # Get class ID
            class_label = results[0].names[class_id]  # Get class label from class ID
            class_label_list.append(class_label)

        if 'NO-Hardhat' in class_label_list and 'NO-Mask' in class_label_list and 'NO-Safety Vest' in class_label_list:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Time stamp to save the image
            temp_file_path = f"temp_{timestamp}.jpg"
            cv2.imwrite(temp_file_path, annotated_frame)  # Save the image
            st.write("Person is not wearing proper PPE")

        # Display the annotated frame in Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame_rgb, channels='RGB')

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
