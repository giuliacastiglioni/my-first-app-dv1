import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

st.title("Analisi Video per il Calcio a 7 Amatoriale")

# Caricamento del video
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Caricare modello YOLOv8
    model = YOLO("yolov8n.pt")
    
    # Lettura del video
    cap = cv2.VideoCapture(video_path)
    
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converti frame in RGB per Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Rilevamento giocatori
        results = model(frame_rgb)
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostra il frame con rilevamento giocatori
        stframe.image(frame_rgb, channels="RGB")
        
    cap.release()
    
    st.success("Video elaborato con successo!")
