import streamlit as st
from ultralytics import YOLO
import cv2
import datetime
import numpy as np

# Set up Streamlit layout
st.title("YOLOv8 Object Detection")
st.sidebar.title("Settings")

# Load YOLOv8 model
model_path = st.sidebar.text_input("Model Path", "yolov8n.pt")
model = YOLO(model_path)

# Load the best weights from trained Model
trained_model_path = st.sidebar.text_input("Trained Model Path", "/path/to/your/best.pt")
trained_detection_model = YOLO(trained_model_path)

# Open the video file - '0' meaning Webcam
cap = cv2.VideoCapture(0)

# Define classes
class_map = {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Streamlit video display
stframe = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = trained_detection_model.track(frame, persist=True, conf=0.4, classes=[0, 1, 2, 3, 4, 5, 6, 7])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        class_label_list = []

        for box in results[0].boxes:  # Iterate over bounding boxes in the frame
            class_id = int(box.cls)  # Get class ID
            class_label = results[0].names[class_id]  # Get class label from class ID
            class_label_list.append(class_label)

        if 'NO-Hardhat' in class_label_list and 'NO-Mask' in class_label_list and 'NO-Safety Vest' in class_label_list:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Time stamp to save the image
            image_name = f"image_{timestamp}.jpg"  # Name the image using timestamp
            cv2.imwrite(image_name, annotated_frame)  # Save the image
            st.warning("Person is not wearing proper PPE")

        # Display the annotated frame
        stframe.image(annotated_frame, channels="BGR")

        # Break the loop if 'q' is pressed
        if st.sidebar.button("Stop"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()
