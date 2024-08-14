import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import datetime

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Load the best weights from the trained model
trained_detection_model = YOLO('best.pt')  # Ensure this path is relative

# Streamlit app
st.title("PPE Detection with YOLOv8")

# Example for testing with a static image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Run YOLOv8 tracking on the frame
    results = trained_detection_model.track(frame, persist=True, conf=0.4, classes=[0,1,2,3,4,5,6,7])

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    class_label_list = []

    for box in results[0].boxes:
        class_id = int(box.cls)
        class_label = results[0].names[class_id]
        class_label_list.append(class_label)

    if 'NO-Hardhat' in class_label_list and 'NO-Mask' in class_label_list and 'NO-Safety Vest' in class_label_list:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        temp_file_path = f"temp_{timestamp}.jpg"
        cv2.imwrite(temp_file_path, annotated_frame)
        st.write("Person is not wearing proper PPE")

    # Display the annotated frame in Streamlit
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_frame_rgb, channels='RGB')
