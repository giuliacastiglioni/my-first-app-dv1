import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque

# ---------------------- STILE PERSONALIZZATO ----------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #1B5E20;
        width: 100%;
    }
    .stButton > button {
        background-color: #FFD700;
        color: black;
        border-radius: 10px;
        font-size: 14px;
    }
    .stTitle {
        font-size: 28px;
        color: white;
        text-shadow: 1px 1px 8px green;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: #3CB371;
        color: white;
        font-size: 14px;
    }
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- FUNZIONI DI SUPPORTO ----------------------
def load_vj_logo():
    """Carica il logo della squadra Vittoria Junior."""
    logo_path = "/workspaces/my-first-app-dv1/data/logo.jpg"
    img = cv2.imread(logo_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_dominant_color(image):
    """Determina il colore dominante (rosso o bianco) di una maglia per identificare la squadra."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width, _ = image.shape
    upper_body = hsv_image[:height//2, :]
    red_mask1 = cv2.inRange(upper_body, np.array([0,70,50]), np.array([10,255,255]))
    red_mask2 = cv2.inRange(upper_body, np.array([170,70,50]), np.array([180,255,255]))
    red_area = np.sum(red_mask1 + red_mask2)
    white_mask = cv2.inRange(upper_body, np.array([0,0,200]), np.array([180,50,255]))
    white_area = np.sum(white_mask)
    area_threshold = (height * width) // 10
    if red_area > white_area and red_area > area_threshold:
        return "VJ"
    elif white_area > red_area and white_area > area_threshold:
        return "Squadra 2"
    else:
        return "Non identificato"
    
# Funzione per determinare il colore prevalente nella maglia
#def get_dominant_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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



def track_ball_with_yolo(frame, model):
    """Rileva la palla nel frame usando YOLO e ne traccia la posizione."""
    results = model(frame)
    min_confidence = 0.5
    ball_position = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == 32 and box.conf[0] > min_confidence:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, "Palla", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                ball_position = ((x1+x2)//2, (y1+y2)//2)
    return frame, ball_position

def detect_goal(ball_position, goal_area):
    """Verifica se la palla è entrata in porta."""
    x, y = ball_position
    x_min, x_max, y_min, y_max = goal_area
    return x_min <= x <= x_max and y_min <= y <= y_max

# ---------------------- STATO DELLA SESSIONE ----------------------
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = None
if 'replay' not in st.session_state:
    st.session_state.replay = False
st.session_state.ball_positions = []
st.session_state.player_positions = {"VJ": [], "Squadra 2": []}
st.session_state.pass_messages = deque(maxlen=50)
st.session_state.contrasts = []
st.session_state.goals = []

# ---------------------- INTERFACCIA PRINCIPALE ----------------------
st.title("**⚽ Video Analysis VJ Open**")
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    team_selected = st.selectbox("Seleziona la squadra da analizzare", ["VJ", "Squadra 2"])
    if team_selected == "VJ":
        st.image(load_vj_logo(), caption="Vittoria Junior", use_container_width=True)

    if st.button("Rivedi il video analizzato"):
        st.session_state.replay = True
        st.session_state.ball_positions = []
        st.session_state.player_positions = {"VJ": [], "Squadra 2": []}
        st.session_state.pass_messages = deque(maxlen=50)
        st.session_state.contrasts = []
        st.session_state.goals = []

    if st.session_state.replay or st.session_state.results_cache is None:
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(video_path)
        stframe_player = st.empty()
        stframe_ball = st.empty()
        player_positions = {"VJ": [], "Squadra 2": []}
        previous_ball_position = None
        previous_player = None
        previous_positions = []
        frame_skip = 3
        frame_counter = 0
        pass_distance_threshold = 30
        ball_move_threshold = 20
        possession_history = deque(maxlen=10)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_positions_vj = []
            current_positions_s2 = []

            results = model(frame_rgb)
            for res in results:
                for box in res.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    player_img = frame[y1:y2, x1:x2]
                    team = get_dominant_color(player_img)
                    center = ((x1+x2)//2, (y1+y2)//2)
                    if team == "VJ":
                        player_positions["VJ"].append(center)
                        current_positions_vj.append(center)
                    elif team == "Squadra 2":
                        player_positions["Squadra 2"].append(center)
                        current_positions_s2.append(center)
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame_rgb, team, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            frame_ball, ball_position = track_ball_with_yolo(frame, model)
            if ball_position:
                st.session_state.ball_positions.append(ball_position)

                current_player = None
                min_dist = float('inf')
                for team, positions in [("VJ", current_positions_vj), ("Squadra 2", current_positions_s2)]:
                    for pos in positions:
                        dist = np.linalg.norm(np.array(ball_position) - np.array(pos))
                        if dist < min_dist:
                            min_dist = dist
                            current_player = team

                possession_history.append(current_player)

                if len(possession_history) == possession_history.maxlen:
                    team_before = possession_history[0]
                    team_after = possession_history[-1]
                    if team_before and team_after and team_before != team_after:
                        if previous_ball_position is not None and np.linalg.norm(np.array(ball_position) - np.array(previous_ball_position)) > 40:
                            st.session_state.pass_messages.append((previous_ball_position, ball_position, team_before, team_after))

                # Contrasto se due giocatori sono vicini alla palla
                all_positions = current_positions_vj + current_positions_s2
                close_players = [pos for pos in all_positions if np.linalg.norm(np.array(ball_position) - np.array(pos)) < 30]
                if len(close_players) >= 2:
                    st.session_state.contrasts.append((ball_position, frame_counter))

                # Rilevamento del goal
                goal_area = (200, 400, 100, 300)
                if detect_goal(ball_position, goal_area):
                    st.session_state.goals.append((ball_position, frame_counter))
                    st.write("GOAL! La palla è entrata in porta.")

                previous_ball_position = ball_position

            stframe_player.image(frame_rgb, channels="RGB")
            stframe_ball.image(frame_ball, channels="RGB")

        cap.release()
        st.session_state.results_cache = player_positions

        # ---------------------- VISUALIZZAZIONE HEATMAP ----------------------
        st.subheader("Heatmap di Movimento dei Giocatori")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        all_y = []
        for i, (team, positions) in enumerate(player_positions.items()):
            if positions:
                pos_arr = np.array(positions)
                h = axs[i].hist2d(pos_arr[:,0], pos_arr[:,1], bins=50, cmap="viridis")
                axs[i].set_title(f"Heatmap {team}")
                axs[i].set_xlabel("Larghezza campo (X)")
                axs[i].set_ylabel("Lunghezza campo (Y)")
                all_y.extend(pos_arr[:,1])
                plt.colorbar(h[3], ax=axs[i])
        if all_y:
            for ax in axs:
                ax.set_ylim(min(all_y), max(all_y))
        st.pyplot(fig)

        # ---------------------- TRAIETTORIA DELLA PALLA ----------------------
        st.subheader("Traiettoria della Palla")
        bp = np.array([pos for pos in st.session_state.ball_positions if isinstance(pos, tuple)])
        if bp.size > 0:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(bp[:,0], bp[:,1], 'ro-', markersize=5)
            ax.set_title("Traiettoria della Palla")
            ax.set_xlabel("Larghezza campo (X)")
            ax.set_ylabel("Lunghezza campo (Y)")
            st.pyplot(fig)

        # ---------------------- MAPPA COMPLETA DEI PASSAGGI ----------------------
        st.subheader("Mappa dei Passaggi Rilevati")
        if st.session_state.pass_messages:
            fig, ax = plt.subplots(figsize=(10,7))
            for i, (start, end, from_team, to_team) in enumerate(st.session_state.pass_messages):
                color = 'blue' if from_team == "VJ" else 'red'
                ax.annotate(f"{i+1}", xy=end, textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color=color)
                ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                         head_width=10, head_length=10, fc=color, ec=color, alpha=0.7)
            ax.set_title("Tutti i passaggi rilevati")
            ax.set_xlabel("Larghezza campo (X)")
            ax.set_ylabel("Lunghezza campo (Y)")
            st.pyplot(fig)

        # ---------------------- CONTRASTI ----------------------
        st.subheader("Contrasti Rilevati")
        if st.session_state.contrasts:
            fig, ax = plt.subplots(figsize=(8,6))
            for pos, frame_id in st.session_state.contrasts:
                ax.plot(pos[0], pos[1], 'kx')
            ax.set_title("Punti di Contrasto")
            ax.set_xlabel("Larghezza campo (X)")
            ax.set_ylabel("Lunghezza campo (Y)")
            st.pyplot(fig)

        # ---------------------- GOAL ----------------------
        st.subheader("Goal Rilevati")
        if st.session_state.goals:
            for i, (pos, frame_id) in enumerate(st.session_state.goals):
                st.write(f"GOAL #{i+1} al frame {frame_id} in posizione {pos}")

 # Dashboard interattiva per le statistiche
        st.subheader("Dashboard Interattiva")
        player_limit = st.slider("Limite giocatori da visualizzare", min_value=1, max_value=14, value=5)
        selected_positions = player_positions[team_selected][:player_limit]
        st.write(f"Posizioni visualizzate per {team_selected}: {len(selected_positions)} giocatori")
        if selected_positions:
            selected_positions = np.array(selected_positions)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(selected_positions[:, 0], selected_positions[:, 1], color='blue', marker='x')
            ax.set_title(f"Posizioni Giocatori - {team_selected}")
            ax.set_xlabel("Posizione X")
            ax.set_ylabel("Posizione Y")
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist2d(selected_positions[:, 0], selected_positions[:, 1], bins=50, cmap="YlGnBu")
            ax.set_title(f"Heatmap Posizioni Giocatori - {team_selected}")
            ax.set_xlabel("Posizione X")
            ax.set_ylabel("Posizione Y")
            st.pyplot(fig)
        

        st.success("Video elaborato con successo!")
    else:
        st.write("Premi il pulsante per rivedere il video analizzato.")
