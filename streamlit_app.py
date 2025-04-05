import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from collections import deque

st.set_page_config(layout="wide")

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
def get_dominant_color(image, k=2):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    dominant_color = kmeans.cluster_centers_.astype(int)
    return dominant_color

def detect_goal(ball_position, goal_area):
    x, y = ball_position
    x_min, x_max, y_min, y_max = goal_area
    return x_min <= x <= x_max and y_min <= y <= y_max

def track_ball_with_yolo(frame, model):
    results = model(frame)
    min_confidence = 0.5
    ball_position = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if int(box.cls[0]) == 32 and box.conf[0] > min_confidence:
                ball_position = ((x1+x2)//2, (y1+y2)//2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, "Palla", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return frame, ball_position

def save_analysis(pass_events, contrast_events, ball_positions):
    df_passaggi = pd.DataFrame(pass_events, columns=["Start", "End", "Team1", "Team2"])
    df_contrasti = pd.DataFrame(contrast_events, columns=["Team1", "Team2", "Position"])
    df_palla = pd.DataFrame(ball_positions, columns=["X", "Y"])

    with pd.ExcelWriter("analysis_output.xlsx") as writer:
        df_passaggi.to_excel(writer, sheet_name="Passaggi")
        df_contrasti.to_excel(writer, sheet_name="Contrasti")
        df_palla.to_excel(writer, sheet_name="Traiettoria Palla")

    st.success("Analisi salvata come 'analysis_output.xlsx'")

# ---------------------- STREAMLIT ----------------------
st.title("**⚽ Video Analysis VJ Open**")
uploaded_file = st.file_uploader("Carica un video", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    team_selected = st.selectbox("Seleziona la squadra da analizzare", ["VJ", "Squadra 2"])
    goal_area = st.slider("Seleziona l'area del gol (X_min, X_max, Y_min, Y_max)", 
                          min_value=0, max_value=1000, value=(200, 400, 100, 300))

    if st.button("Avvia analisi video"):
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        ball_positions = []
        player_positions = {"VJ": [], "Squadra 2": []}
        pass_events = []
        contrast_events = []
        previous_ball_position = None

        possession_history = deque(maxlen=10)
        previous_player = None

        frame_skip = 2
        frame_counter = 0

        # Barra di progresso
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # Aggiorna la barra di progresso ogni 10 frame
            if frame_counter % 10 == 0:
                progress = frame_counter / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100
                progress_bar.progress(progress)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            current_positions_vj = []
            current_positions_s2 = []

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

            frame_ball, ball_position = track_ball_with_yolo(frame, model)
            if ball_position:
                ball_positions.append(ball_position)

                # Nuova logica PASSAGGI
                current_player = None
                min_dist = float('inf')
                for team, positions in [("VJ", current_positions_vj), ("Squadra 2", current_positions_s2)]:
                    for pos in positions:
                        dist = np.linalg.norm(np.array(ball_position) - np.array(pos))
                        if dist < min_dist:
                            min_dist = dist
                            current_player = team

                if current_player != previous_player and previous_player is not None:
                    if previous_ball_position is not None:
                        movement = np.linalg.norm(np.array(ball_position) - np.array(previous_ball_position))
                        if movement > 30:
                            pass_events.append((previous_ball_position, ball_position, previous_player, current_player))

                if current_player != previous_player and current_player is not None and previous_player is not None:
                    contrast_events.append((previous_player, current_player, ball_position))

                previous_player = current_player
                previous_ball_position = ball_position

                if detect_goal(ball_position, goal_area):
                    st.warning("GOAL! La palla è entrata in porta.")

            stframe.image(frame_ball, channels="RGB")

        cap.release()

        # VISUALIZZAZIONE
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Traiettoria della Palla")
            bp = np.array([pos for pos in ball_positions if isinstance(pos, tuple)])
            if bp.size > 0:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.plot(bp[:,0], bp[:,1], 'ro-', markersize=5)
                ax.set_title("Traiettoria della Palla")
                ax.set_xlabel("Larghezza campo (X)")
                ax.set_ylabel("Lunghezza campo (Y)")
                st.pyplot(fig)

        with col2:
            st.subheader("Passaggi Rilevati")
            for start, end, team1, team2 in pass_events:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', lw=2)
                ax.set_title(f"Passaggio da {team1} a {team2}")
                st.pyplot(fig)

            st.subheader("Contrasti Rilevati")
            for team1, team2, pos in contrast_events:
                st.write(f"Contrasto tra {team1} e {team2} a posizione {pos}")

        # CREAZIONE HEATMAP CALCIATRICI
        st.subheader("Heatmap delle Calciatrici")
        all_positions = player_positions["VJ"] + player_positions["Squadra 2"]
        all_positions = np.array(all_positions)
        if all_positions.size > 0:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.imshow(np.zeros_like(frame), cmap="gray", alpha=0.5)
            heatmap = np.histogram2d(all_positions[:, 1], all_positions[:, 0], bins=(30, 30), range=[[0, frame.shape[0]], [0, frame.shape[1]]])[0]
            ax.imshow(heatmap.T, origin="lower", cmap="hot", interpolation="nearest", alpha=0.7)
            ax.set_title("Heatmap delle Calciatrici in Campo")
            st.pyplot(fig)

        save_analysis(pass_events, contrast_events, ball_positions)

        st.success("Analisi completata!")
