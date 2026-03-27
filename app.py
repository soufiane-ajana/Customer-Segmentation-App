import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# 1. CONFIGURATION AVANCÉE DU DASHBOARD
st.set_page_config(page_title="Customer Insights Pro", page_icon="🌌", layout="wide")

# --- INJECTION DE CSS POUR LE DESIGN (Animations et Couleurs) ---
st.markdown("""
    <style>
    /* Design des boutons */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
    }
    /* Style des métriques */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #4b6cb7;
    }
    </style>
""", unsafe_allow_html=True)

# 2. EN-TÊTE DU DASHBOARD
st.title("🌌 Plateforme d'Intelligence Client (V2.0)")
st.markdown("*Système de classification prédictive propulsé par Machine Learning.*")
st.markdown("---")

# 3. CHARGEMENT DU MODÈLE ET DES DONNÉES HISTORIQUES
@st.cache_resource
def charger_modele():
    with open('modele_segmentation.pkl', 'rb') as fichier:
        return pickle.load(fichier)

@st.cache_data
def charger_donnees():
    # On charge l'historique pour le visuel
    return pd.read_csv('Resultat_Segmentation.csv')

modele = charger_modele()
df_historique = charger_donnees()

# 4. STRUCTURE EN COLONNES (Gauche: Contrôles / Droite: Résultats)
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("🎛️ Profil du Prospect")
    st.info("Ajustez les paramètres en temps réel pour simuler un profil et déclencher l'algorithme de classification.")
    
    # Sliders plus design
    revenu = st.slider("💰 Revenu Annuel estimé (k$)", min_value=10, max_value=150, value=60, step=1)
    score = st.slider("🔥 Score d'engagement (1-100)", min_value=1, max_value=100, value=50, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True) # Espace
    
    analyser = st.button("🚀 LANCER L'ANALYSE PRÉDICTIVE")

with col2:
    st.header("📊 Diagnostic et Cartographie")
    
    # --- Création d'un graphique Plotly interactif en arrière-plan ---
    # On map les couleurs pour faire joli
    couleurs_clusters = {1: '#3498db', 2: '#f1c40f', 3: '#e74c3c', 4: '#9b59b6', 5: '#2ecc71'}
    df_historique['Couleur'] = df_historique['Cluster'].map(couleurs_clusters)
    
    fig = px.scatter(df_historique, 
                     x='Annual Income (k$)', 
                     y='Spending Score (1-100)', 
                     color='Cluster',
                     title="Cartographie de la base client actuelle",
                     template="plotly_dark", # Thème sombre très élégant
                     hover_data=['Age'])
    
    fig.update_traces(marker=dict(size=10, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
    
    # 5. ACTION LORS DU CLIC SUR LE BOUTON
    if analyser:
        # Prédiction
        donnees_prospect = pd.DataFrame({'Annual Income (k$)': [revenu], 'Spending Score (1-100)': [score]})
        prediction = modele.predict(donnees_prospect)[0]
        
        # Ajout du point du nouveau prospect sur le graphique (Animation visuelle)
        fig.add_scatter(x=[revenu], y=[score], mode='markers+text', 
                        marker=dict(color='white', size=25, symbol='star', line=dict(color='red', width=2)),
                        name="CIBLE ACTUELLE",
                        text=["VOUS ÊTES ICI"], textposition="top center")
        
        # Affichage du graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage des résultats sous forme de "Cards" (Métriques)
        st.subheader("🎯 Résultat de la classification")
        
        # On crée 3 petites colonnes pour les KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Revenu Saisi", f"{revenu} k$")
        kpi2.metric("Score Saisi", f"{score} pts")
        kpi3.metric("Groupe Prédit", f"Cluster n°{prediction}")
        
        # Recommandations stratégiques avec animations Streamlit
        if prediction == 2:
            st.success("### 💎 PROFIL IDENTIFIÉ : VIP (Haut potentiel)")
            st.write("Action recommandée : Déployer le service Conciergerie Premium. Ce client a un fort pouvoir d'achat et une forte propension à dépenser.")
            st.balloons() # Petite animation Streamlit sympa !
            
        elif prediction == 4:
            st.warning("### ⚠️ PROFIL IDENTIFIÉ : Potentiel Inexploité")
            st.write("Action recommandée : Stratégie de réassurance. Le budget est présent, mais l'offre actuelle ne déclenche pas l'achat.")
            
        elif prediction == 3:
            st.error("### 🔥 PROFIL IDENTIFIÉ : Les Insouciants")
            st.write("Action recommandée : Push Marketing direct. Forte sensibilité aux tendances, mais attention au risque d'impayé.")
            
        elif prediction == 1:
            st.info("### 📊 PROFIL IDENTIFIÉ : Classe Moyenne (Cœur de cible)")
            st.write("Action recommandée : Intégration au programme de fidélité standard pour maintenir l'engagement sur la durée.")
            
        else:
            st.markdown("### 💰 PROFIL IDENTIFIÉ : Les Économes")
            st.write("Action recommandée : Approche agressive sur le prix (Promotions, destockage). Très faible élasticité prix.")
            
    else:
        # Si on n'a pas encore cliqué, on affiche juste le graphique de base
        st.plotly_chart(fig, use_container_width=True)