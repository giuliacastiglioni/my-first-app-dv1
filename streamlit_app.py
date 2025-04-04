import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

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
        return "Squadra 1 (Rosso)"  # Bianco con rosso
    elif yellow_area > red_area and yellow_area > blue_area:
        return "Squadra 2 (Giallo)"  # Giallo
    elif blue_area > red_area and blue_area > yellow_area:
        return "Squadra 2 (Blu)"  # Blu
    else:
        return "Non identificato"

# Funzione per determinare le azioni (dribbling, passaggio, tiro)
def detect_actions(player_positions, ball_position):
    actions = []
    
    for player in player_positions:
        x, y = player
        
        # Esempio di regola per un passaggio: due giocatori vicini tra loro
        if len(player_positions) > 1:  # Se ci sono più giocatori
            for other_player in player_positions:
                if player != other_player:
                    dist = np.linalg.norm(np.array(player) - np.array(other_player))
                    if dist < 50:  # Se la distanza tra i giocatori è bassa, è probabile un passaggio
                        actions.append("Passaggio")
        
        # Esempio di regola per un dribbling: movimento rapido del giocatore
        if len(player_positions) > 2:  # Movimenti rapidi tra i frame
            actions.append("Dribbling")
        
        # Esempio di regola per un tiro: se il giocatore si sposta verso la porta
        if ball_position is not None:
            if y < 50:  # Il giocatore è vicino alla porta (ipotetica posizione della porta)
                actions.append("Tiro")
    
    return actions

# Funzione per ottenere la posizione della palla (sintetico, come esempio)
def get_ball_position(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])  # Gamma per il colore arancione (può variare)
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return int(x), int(y)
    return None

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Caricare modello YOLOv8
    model = YOLO("yolov8n.pt")
    
    # Lettura del video
    cap = cv2.VideoCapture(video_path)
    
    stframe = st.empty()
    player_positions = []  # Lista per memorizzare le posizioni dei giocatori
    ball_position = None  # Posizione della palla
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converti frame in RGB per Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Rilevamento giocatori
        results = model(frame_rgb)
        
        # Rilevamento dei giocatori e posizioni
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                player_center_x = (x1 + x2) // 2
                player_center_y = (y1 + y2) // 2
                player_positions.append((player_center_x, player_center_y))
                
                # Estrai la parte dell'immagine relativa al giocatore
                player_img = frame[y1:y2, x1:x2]
                
                # Determina il colore dominante nella maglia
                team = get_dominant_color(player_img)
                
                # Disegna rettangolo, ID e squadra
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{team}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Rilevamento della palla
        ball_position = get_ball_position(frame)
        if ball_position:
            cv2.circle(frame_rgb, ball_position, 10, (0, 0, 255), -1)  # Disegna la palla
        
        # Rileva azioni importanti (dribbling, passaggi, tiri)
        actions = detect_actions(player_positions, ball_position)
        
        # Mostra azioni nel video
        for action in actions:
            cv2.putText(frame_rgb, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Mostra il frame
        stframe.image(frame_rgb, channels="RGB")
        
        # Resetta la lista delle posizioni per il prossimo frame
        player_positions = []

    cap.release()
    
    st.success("Video elaborato con successo!")
