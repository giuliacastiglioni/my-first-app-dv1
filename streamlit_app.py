import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

st.title("Analisi Video per il Calcio a 7 Femminile")

# Caricamento del video
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

# Funzione per determinare il colore prevalente in HSV
def get_dominant_color_hsv(image):
    # Converti l'immagine da BGR a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv_image, axis=(0, 1))
    return avg_color

# Funzione per determinare la squadra in base al colore in HSV
def assign_team_color_hsv(hsv_color):
    h, s, v = hsv_color

    # Squadra 1: Bianco con toni di rosso (in HSV: Hue intorno a 0-10, alta saturazione, alta luminosità)
    if 0 <= h <= 10 and 150 <= s <= 255 and 180 <= v <= 255:  # Rosso-bianco
        return "Squadra 1"
    # Squadra 2: Giallo (Hue intorno a 20-40, alta saturazione, alta luminosità)
    elif 20 <= h <= 40 and 150 <= s <= 255 and 150 <= v <= 255:  # Giallo
        return "Squadra 2"
    # Squadra 2: Blu (Hue intorno a 100-130, alta saturazione, alta luminosità)
    elif 100 <= h <= 130 and 150 <= s <= 255 and 150 <= v <= 255:  # Blu
        return "Squadra 2"
    else:
        return "Non identificato"

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
                
                # Estrai la parte dell'immagine relativa al giocatore
                player_img = frame[y1:y2, x1:x2]
                
                # Calcola il colore dominante in HSV
                dominant_color_hsv = get_dominant_color_hsv(player_img)
                
                # Assegna la squadra in base al colore
                team = assign_team_color_hsv(dominant_color_hsv)
                
                # Disegna rettangolo, ID e squadra
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"Squadra: {team}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Mostra il frame con il rilevamento dei giocatori e la distinzione della squadra
        stframe.image(frame_rgb, channels="RGB")
    
    cap.release()
    
    st.success("Video elaborato con successo!")
