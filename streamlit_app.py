import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

st.title("Analisi Video per il Calcio a 7 Amatoriale")

# Caricamento del video
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Lettura del video
    cap = cv2.VideoCapture(video_path)
    
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converti frame in RGB per Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Mostra il frame
        stframe.image(frame, channels="RGB")
        
    cap.release()
    
    st.success("Video elaborato con successo!")
