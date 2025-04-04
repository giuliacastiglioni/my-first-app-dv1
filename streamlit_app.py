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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Applichiamo una maschera per il colore rosso (bianco-rosso), giallo e blu
    lower_red = np.array([0, 50, 100])
    upper_red = np.array([10, 255, 255])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    blue_area = np.sum(blue_mask)
    
    if red_area > yellow_area and red_area > blue_area:
        return "VJ"
    elif yellow_area > red_area and yellow_area > blue_area:
        return "Squadra 2"
    elif blue_area > red_area and blue_area > yellow_area:
        return "Squadra 2"
    else:
        return "Non identificato"

# Variabile per memorizzare lo stato del video e dei risultati
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = None
    st.session_state.replay = False

# Caricamento e analisi del video
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    if st.button('Rivedi il video analizzato'):
        st.session_state.replay = True
        st.session_state.results_cache = None  # Svuota la cache per una nuova analisi
    
    # Se il video è già stato analizzato, carica dalla cache
    if st.session_state.replay or st.session_state.results_cache is None:
        # Carica modello YOLOv8
        model = YOLO("yolov8n.pt")
    
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        player_positions = {"VJ": [], "Squadra 2": []}
        
        frame_skip = 3  # Analizza ogni 3 frame per un buon compromesso tra velocità e precisione
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # Usa la risoluzione originale per il rilevamento
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Rilevamento dei giocatori con YOLO
            results = model(frame_rgb)
            
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    player_img = frame[y1:y2, x1:x2]
                    
                    # Determina il colore prevalente nella maglia
                    team = get_dominant_color(player_img)
                    
                    # Coordinate centrali del giocatore per tenere traccia della sua posizione
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if team == "VJ":
                        player_positions["VJ"].append((center_x, center_y))
                    elif team == "Squadra 2":
                        player_positions["Squadra 2"].append((center_x, center_y))
                    
                    # Disegna il rettangolo attorno al giocatore e il nome della squadra
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f"{team}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Visualizza il frame corrente
            stframe.image(frame_rgb, channels="RGB")
        
        cap.release()

        # Memorizza i risultati per il replay
        st.session_state.results_cache = player_positions
        
        # Visualizzazione Heatmap (o altro tipo di analisi)
        st.subheader("Heatmap di Movimento dei Giocatori")
        
        # Prepara i dati per la visualizzazione
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imposta una scala comune per gli assi X e Y
        x_limit = (0, 1920)  # Risoluzione orizzontale
        y_limit = (0, 1080)  # Risoluzione verticale
        
        for i, (team, positions) in enumerate(player_positions.items()):
            if positions:
                positions = np.array(positions)
                axs[i].hist2d(positions[:, 0], positions[:, 1], bins=50, cmap="YlGnBu")
                axs[i].set_title(f"Heatmap {team}")
                axs[i].set_xlabel("Posizione X")
                axs[i].set_ylabel("Posizione Y")
                
                # Imposta la stessa scala per entrambi i grafici
                axs[i].set_xlim(x_limit)
                axs[i].set_ylim(y_limit)
        
        st.pyplot(fig)
        
        st.success("Video elaborato con successo!")

    else:
        st.write("Premi il pulsante per rivedere il video analizzato.")
