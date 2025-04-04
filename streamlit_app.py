import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import trackpy as tp

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

# Funzione per tracciare la palla usando trackpy
def track_ball_with_trackpy(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Imposta la maschera per il colore giallo (palla)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    
    # Filtro per ridurre il rumore
    yellow_mask = cv2.GaussianBlur(yellow_mask, (15, 15), 0)
    
    # Trova i contorni della palla
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_positions = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtro per dimensione della palla
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:  # Filtro per dimensione minima della palla
                ball_positions.append([x, y, radius])
                
    return ball_positions

# Variabile per memorizzare lo stato del video e dei risultati
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = None
    st.session_state.replay = False

if 'ball_positions' not in st.session_state:  # Inizializza ball_positions
    st.session_state.ball_positions = []

if 'player_positions' not in st.session_state:  # Inizializza player_positions
    st.session_state.player_positions = {"VJ": [], "Squadra 2": []}

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
        stframe_player = st.empty()
        stframe_ball = st.empty()

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

            # Traccia la palla e aggiorna la sua posizione
            ball_positions = track_ball_with_trackpy(frame)
            for (x, y, radius) in ball_positions:
                st.session_state.ball_positions.append([x, y])

            # Visualizza il frame con il tracking dei giocatori
            stframe_player.image(frame_rgb, channels="RGB")

            # Visualizza il frame con il tracking della palla
            frame_ball = frame.copy()
            for (x, y, radius) in ball_positions:
                cv2.circle(frame_ball, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame_ball, (int(x), int(y)), 5, (0, 0, 255), -1)  # Centro della palla
            stframe_ball.image(frame_ball, channels="RGB")
        
        cap.release()

        # Memorizza i risultati per il replay
        st.session_state.results_cache = player_positions
        
        # Visualizzazione Heatmap (o altro tipo di analisi)
        st.subheader("Heatmap di Movimento dei Giocatori")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for i, (team, positions) in enumerate(player_positions.items()):
            if positions:
                positions = np.array(positions)
                axs[i].hist2d(positions[:, 0], positions[:, 1], bins=50, cmap="YlGnBu")
                axs[i].set_title(f"Heatmap {team}")
                axs[i].set_xlabel("Posizione X")
                axs[i].set_ylabel("Posizione Y")
        st.pyplot(fig)  # Passiamo la figura a st.pyplot()

        # Visualizzazione della traiettoria della palla separata dalla heatmap
        st.subheader("Traiettoria della Palla")
        ball_positions = np.array(st.session_state.ball_positions)
        if ball_positions.size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))  # Creiamo una nuova figura
            ax.plot(ball_positions[:, 0], ball_positions[:, 1], marker='o', markersize=5, color='red')
            ax.set_title("Traiettoria della Palla")
            ax.set_xlabel("Posizione X")
            ax.set_ylabel("Posizione Y")
            st.pyplot(fig)  # Passiamo la figura a st.pyplot()

        st.success("Video elaborato con successo!")

    else:
        st.write("Premi il pulsante per rivedere il video analizzato.")
