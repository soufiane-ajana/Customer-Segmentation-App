import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# 1. CONFIGURATION AVANCÉE DU DASHBOARD
st.set_page_config(page_title="Customer Insights Pro", page_icon="🌌", layout="wide")

st.markdown("""
    <style>
    /* 1. FOND GLOBAL SOMBRE ET ÉLÉGANT */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #2b2b2b 0%, #0E1117 70%);
        color: #FAFAFA;
    }

    /* 2. ANIMATION DES CARTES DE RÉSULTATS (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out; /* C'est ça qui crée la fluidité */
        border: 1px solid #333;
    }
    /* L'effet 'Hover' (survol de la souris) sur les KPIs */
    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02); /* Soulèvement et léger zoom */
        box-shadow: 0 12px 25px rgba(255,255,255,0.1); /* Ombre blanche diffuse */
        border-color: #4b6cb7; /* Bordure qui s'allume en bleu */
    }

    /* 3. ANIMATION DU BOUTON PRINCIPAL */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: bold;
        transition: all 0.4s ease;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0px 10px 25px rgba(75, 108, 183, 0.6); /* Halo lumineux bleu */
        letter-spacing: 1px; /* Le texte s'écarte légèrement */
    }

    /* 4. ANIMATION DES ONGLETS (TABS) */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-3px);
        color: #4b6cb7;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 Plateforme d'Intelligence Client")
st.markdown("*Système de classification prédictive propulsé par Machine Learning.*")
st.markdown("---")

# 2. CHARGEMENT
@st.cache_resource
def charger_modele():
    with open('modele_segmentation.pkl', 'rb') as fichier:
        return pickle.load(fichier)

@st.cache_data
def charger_donnees():
    return pd.read_csv('Resultat_Segmentation.csv')

modele = charger_modele()
df_historique = charger_donnees()

# 3. STRUCTURE EN COLONNES
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("🎛️ Profil du Prospect")
    st.info("Ajustez les paramètres en temps réel.")
    revenu = st.slider("💰 Revenu Annuel (k$)", min_value=10, max_value=150, value=60, step=1)
    score = st.slider("🔥 Score d'engagement (1-100)", min_value=1, max_value=100, value=50, step=1)
    st.markdown("<br>", unsafe_allow_html=True)
    analyser = st.button("🚀 LANCER L'ANALYSE PRÉDICTIVE")

with col2:
    # --- NOUVEAUTÉ : LES ONGLETS (TABS) ---
    tab_graphique, tab_explications = st.tabs(["📊 Diagnostic en direct", "📖 Dictionnaire des Profils"])
    
    # ONGLET 1 : LE GRAPHIQUE ET LES RÉSULTATS
    with tab_graphique:
        couleurs_clusters = {1: '#3498db', 2: '#f1c40f', 3: '#e74c3c', 4: '#9b59b6', 5: '#2ecc71'}
        df_historique['Couleur'] = df_historique['Cluster'].map(couleurs_clusters)
        
        fig = px.scatter(df_historique, x='Annual Income (k$)', y='Spending Score (1-100)', 
                         color='Cluster', title="Cartographie de la base client actuelle",
                         template="plotly_dark", hover_data=['Age'])
        fig.update_traces(marker=dict(size=10, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
        
        if analyser:
            donnees_prospect = pd.DataFrame({'Annual Income (k$)': [revenu], 'Spending Score (1-100)': [score]})
            prediction = modele.predict(donnees_prospect)[0]
            
            fig.add_scatter(x=[revenu], y=[score], mode='markers+text', 
                            marker=dict(color='white', size=25, symbol='star', line=dict(color='red', width=2)),
                            name="CIBLE ACTUELLE", text=["VOUS ÊTES ICI"], textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Résultat de la classification")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Revenu Saisi", f"{revenu} k$")
            kpi2.metric("Score Saisi", f"{score} pts")
            kpi3.metric("Groupe Prédit", f"Cluster n°{prediction}")
            
            if prediction == 2:
                st.success("### 💎 PROFIL IDENTIFIÉ : VIP (Haut potentiel)")
            elif prediction == 4:
                st.warning("### ⚠️ PROFIL IDENTIFIÉ : Potentiel Inexploité")
            elif prediction == 3:
                st.error("### 🔥 PROFIL IDENTIFIÉ : Les Insouciants")
            elif prediction == 1:
                st.info("### 📊 PROFIL IDENTIFIÉ : Classe Moyenne (Cœur de cible)")
            else:
                st.markdown("### 💰 PROFIL IDENTIFIÉ : Les Économes")
                
        else:
            st.plotly_chart(fig, use_container_width=True)

    # ONGLET 2 : LE DICTIONNAIRE EXPLICATIF
    with tab_explications:
        st.header("Comprendre notre typologie client")
        st.write("L'Intelligence Artificielle a segmenté notre base en 5 comportements types. Voici comment interpréter chaque profil :")
        
        st.success("**💎 Cluster 2 : Les VIP**\n\n* **Profil :** Revenus élevés et dépenses élevées.\n* **Signification :** Ce sont nos clients les plus rentables et fidèles. Ils ne regardent pas à la dépense si la qualité est là.\n* **Stratégie :** Fidélisation premium, offres exclusives, service client prioritaire.")
        
        st.info("**📊 Cluster 1 : La Classe Moyenne**\n\n* **Profil :** Revenus moyens et dépenses moyennes.\n* **Signification :** Le cœur de notre base de données. Ils ont un comportement d'achat rationnel et stable.\n* **Stratégie :** Maintien de l'engagement via des newsletters régulières et un programme de fidélité classique.")
        
        st.error("**🔥 Cluster 3 : Les Insouciants**\n\n* **Profil :** Faibles revenus mais dépenses très élevées.\n* **Signification :** Souvent une population jeune, très sensible aux modes et aux achats d'impulsion, quitte à dépasser leur budget.\n* **Stratégie :** Promotions flash, marketing émotionnel, facilités de paiement.")
        
        st.warning("**⚠️ Cluster 4 : Potentiel Inexploité**\n\n* **Profil :** Forts revenus mais dépenses très faibles.\n* **Signification :** Ils ont l'argent, mais ne l'utilisent pas chez nous. C'est notre plus grand réservoir de croissance.\n* **Stratégie :** Enquêtes de satisfaction, réassurance, offres d'essai pour les convaincre de la valeur de nos services.")
        
        st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #2e3b4e;'><b>💰 Cluster 5 : Les Économes</b><br><br><ul><li><b>Profil :</b> Faibles revenus et faibles dépenses.</li><li><b>Signification :</b> Des clients très prudents, qui chassent la bonne affaire et achètent le strict minimum.</li><li><b>Stratégie :</b> Ne pas investir de gros budgets marketing sur eux. Leur proposer les offres d'entrée de gamme ou de déstockage.</li></ul></div>", unsafe_allow_html=True)
