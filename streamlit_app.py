import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

st.title("Analisi Video per il Calcio a 7 Femminile")

# Caricamento del video
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

# Funzione per determinare il colore prevalente nella maglia
def get_dominant_color(image):
    # Convertiamo l'immagine in HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Applichiamo una maschera per il colore rosso (bianco-rosso), giallo e blu
    lower_red = np.array([0, 50, 100])
    upper_red = np.array([10, 255, 255])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Creiamo le maschere per i vari colori
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Sommiamo le aree di ciascun colore
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    blue_area = np.sum(blue_mask)
    
    # Determiniamo quale colore è il predominante
    if red_area > yellow_area and red_area > blue_area:
        return "VJ"  # Bianco con rosso
    elif yellow_area > red_area and yellow_area > blue_area:
        return "Squadra 2"  # Giallo
    elif blue_area > red_area and blue_area > yellow_area:
        return "Squadra 2"  # Blu
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
                
                # Determina il colore dominante nella maglia
                team = get_dominant_color(player_img)
                
                # Disegna rettangolo, ID e squadra
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{team}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Mostra il frame con il rilevamento dei giocatori e la distinzione della squadra
        stframe.image(frame_rgb, channels="RGB")
    
    cap.release()
    
    st.success("Video elaborato con successo!")
# Funzione per ottenere le coordinate relative dei giocatori
def get_player_position(x1, y1, x2, y2, frame_width):
    # Restituisce la posizione orizzontale (x) relativa del giocatore
    player_center_x = (x1 + x2) / 2
    relative_position_x = player_center_x / frame_width  # Normalizza la posizione
    return relative_position_x

# Inizializzazione della heatmap (100 bin per la larghezza del campo)
heatmap = np.zeros(100)  # Per una rappresentazione semplificata del campo, 100 "bin" orizzontali

# Supponiamo di avere una lista di coordinate di giocatori rilevati
# Esempio di posizione (x1, y1, x2, y2) per alcuni giocatori (queste vengono da YOLO)
positions = [(100, 200, 150, 250), (300, 180, 350, 230), (600, 150, 650, 200)]  # (x1, y1, x2, y2)

# Supponiamo che la larghezza del frame sia 800 (adatta per il tuo video)
frame_width = 800

# Calcola le posizioni relative dei giocatori e aggiorna la heatmap
for (x1, y1, x2, y2) in positions:
    relative_position = get_player_position(x1, y1, x2, y2, frame_width)
    heatmap[int(relative_position * 100)] += 1  # Incrementa la zona corrispondente

# Visualizzazione della heatmap
plt.imshow([heatmap], cmap='hot', aspect='auto')
plt.colorbar(label="Numero di Giocatori")
plt.title("Heatmap delle Posizioni dei Giocatori")
plt.show()