import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Court

# Titolo dell'app
st.title("🏀 Statistiche Basket Femminile College")

# Caricamento dei dati
uploaded_file = st.file_uploader("Carica un file CSV con le statistiche", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Anteprima dei dati:")
    st.dataframe(df.head())
    
    # Selezione della squadra
    if 'Team' in df.columns:
        team_selected = st.selectbox("Seleziona una squadra", df['Team'].unique())
        df_team = df[df['Team'] == team_selected]
        
        # Grafico punti per partita
        if 'Points' in df.columns and 'Game' in df.columns:
            fig, ax = plt.subplots()
            ax.plot(df_team['Game'], df_team['Points'], marker='o', linestyle='-')
            ax.set_title(f"Punti per partita - {team_selected}")
            ax.set_xlabel("Partita")
            ax.set_ylabel("Punti")
            st.pyplot(fig)
        
        # Istogramma delle statistiche
        numeric_columns = df_team.select_dtypes(include=['number']).columns
        stat_selected = st.selectbox("Seleziona una statistica", numeric_columns)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df_team[stat_selected], bins=20, color='blue', alpha=0.7)
        ax_hist.set_title(f"Distribuzione di {stat_selected}")
        ax_hist.set_xlabel(stat_selected)
        ax_hist.set_ylabel("Frequenza")
        st.pyplot(fig_hist)
        
        # Selezione della giocatrice per visualizzare i tiri
        if {'X', 'Y', 'Player', 'Made'}.issubset(df.columns):
            player_selected = st.selectbox("Seleziona una giocatrice", df_team['Player'].unique())
            df_player = df_team[df_team['Player'] == player_selected]
            
            # Creazione del campo da basket
            fig_court, ax_court = plt.subplots(figsize=(5, 4))
            court = Court(court_dims=(28, 15), court_lines=True)  # Misure in metri
            court.draw(ax=ax_court)
            
            # Plot dei tiri
            colors = df_player['Made'].map({1: 'green', 0: 'red'})
            ax_court.scatter(df_player['X'], df_player['Y'], c=colors, edgecolors='black')
            
            ax_court.set_title(f"Tiri di {player_selected}")
            st.pyplot(fig_court)
else:
    st.write("Carica un file CSV per iniziare!")

# Footer
st.write("App creata con ❤️ usando Streamlit e Matplotlib")
