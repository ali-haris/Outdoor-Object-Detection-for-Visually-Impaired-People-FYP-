import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time
import random
from gtts import gTTS
import os
import threading
import base64
import uuid  # For generating unique filenames

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load YOLO model
model = YOLO(r"best.pt")  # Update with your model path

# Define color codes for objects
colors = {
    'orange': (255, 178, 29),
    'pineapple': (207, 210, 49),
    'person': (0, 255, 0)
}

# Stop signal flag
stop_signal = False

# Function to process an image and annotate it
def process_image(image, select_confidence, enable_sound):
    global stop_signal
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image)
    annotated_image = annotate_image(image, results, select_confidence)

    if enable_sound and not stop_signal:
        detected_classes = [
            model.names[int(box.cls[0])]
            for result in results
            for box in result.boxes
            if box.conf[0].item() * 100 > select_confidence
        ]
        if detected_classes:
            threading.Thread(target=speak_detected_objects, args=(detected_classes,)).start()

    return annotated_image, results

# Function to annotate image
def annotate_image(image, results, select_confidence):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0].item() * 100

            if confidence > select_confidence:
                color = colors.get(class_name.lower(), (0, 255, 0))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}%"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Function to process video frames in real-time
def process_video(select_confidence, enable_sound, video_path=None):
    global stop_signal
    stop_signal = False
    cap = cv2.VideoCapture(video_path if video_path else 0)
    stframe = st.empty()
    frame_count = 0
    spoken_objects = set()

    while cap.isOpened():
        if stop_signal:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Process every 200th frame
        if frame_count % 150 == 0:
            annotated_frame, results = process_image(frame, select_confidence, enable_sound)

            # Speech feedback for new objects only
            if enable_sound:
                detected_classes = [
                    model.names[int(box.cls[0])]
                    for result in results
                    for box in result.boxes
                    if box.conf[0].item() * 100 > select_confidence
                ]
                for obj in detected_classes:
                    if obj not in spoken_objects:
                        threading.Thread(target=speak_detected_objects, args=([obj],)).start()
                        spoken_objects.add(obj)

            stframe.image(annotated_frame, channels="BGR")

        frame_count += 1
        time.sleep(0.01)  # Smooth processing

    cap.release()

# Function to convert detected objects to speech
def speak_detected_objects(detected_classes):
    object_counts = {}
    for obj in detected_classes:
        object_counts[obj] = object_counts.get(obj, 0) + 1

    messages = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in object_counts.items()]
    message = ", and ".join(messages) + " detected."

    unique_filename = f"output_{uuid.uuid4()}.mp3"  # Unique file for each speech
    tts = gTTS(text=message, lang='en')
    tts.save(unique_filename)
    os.system(f"start {unique_filename}")  # Play the file
    time.sleep(1)  # Allow time for playback
    os.remove(unique_filename)  # Clean up after playback

# Sidebar stop button logic
def stop_processing():
    global stop_signal
    stop_signal = True

# Streamlit app title and sidebar options
st.title("Outdoor Object Detection for Visually Impaired People")

select_confidence = st.sidebar.slider('Choose Confidence', min_value=10, max_value=100, value=50)
enable_sound = st.sidebar.checkbox("Enable Sound Alert", value=True)

# Stop button
if st.sidebar.button("Stop Video"):
    stop_processing()

# Image upload and processing
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    annotated_image, results = process_image(image, select_confidence, enable_sound)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)

# Video upload and processing
uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    if st.button('Process Uploaded Video'):
        process_video(select_confidence, enable_sound, video_path)

# Start live webcam processing
if st.sidebar.button('Start Live Video'):
    process_video(select_confidence, enable_sound)




import base64
def sidebar_bg(side_bg):

    side_bg_ext = './Blind.jpg'

    st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
    background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
    }}
    </style>
    """,
    unsafe_allow_html=True,
    )
sidebar_bg('./light-bg.jpg')

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSVS6b2FbkSD4zTpNwZuqUPw2KSwzLjcGqA9Cb_9hMeg&s");
    # background-size: 100px 180px;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    # padding-top:100px!important;
    margin-top:50px!important;
    background-repeat: no-repeat;
}
#out-door-object-detection-for-visually-impaired-people-prototype{
    margin-top:-100px!important;}

</style>
"""
# Set the background image using st.markdown

st.markdown(background_image, unsafe_allow_html=True)
result_section_css = '''
<style>
.result-section {
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
'''