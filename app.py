import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# 1. CONFIGURATION AVANCÉE
st.set_page_config(page_title="Customer Insights Pro", page_icon="🌌", layout="wide")

# --- INJECTION DE CSS (Correction du texte invisible et ajout des animations) ---
st.markdown("""
    <style>
    /* FOND GLOBAL SOMBRE */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #0E1117 70%);
    }
    
    /* CORRECTION DU TEXTE INVISIBLE (Forcer la couleur claire) */
    h1, h2, h3, h4, h5, h6, p, li, span, label, .stMarkdown {
        color: #F0F2F6 !important;
    }

    /* ANIMATION DES CARTES DE RÉSULTATS (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
        border: 1px solid #333;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 25px rgba(255,255,255,0.1);
        border-color: #4b6cb7;
    }
    
    /* COULEUR DES VALEURS DES METRIQUES (Les gros chiffres) */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #4b6cb7 !important;
    }

    /* ANIMATION DU BOUTON PRINCIPAL */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white !important;
        font-weight: bold;
        transition: all 0.4s ease;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-4px);
        box-shadow: 0px 10px 25px rgba(75, 108, 183, 0.6);
        letter-spacing: 1px;
    }

    /* ANIMATION DES ONGLETS */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        color: #A0AEC0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-3px);
        color: #4b6cb7 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. EN-TÊTE DU DASHBOARD
st.title("🌌 Plateforme d'Intelligence Client (V3.0)")
st.markdown("*Système de classification prédictive propulsé par Machine Learning pour l'optimisation des stratégies d'acquisition et de fidélisation.*")
st.markdown("---")

# 3. CHARGEMENT DU MODÈLE ET DES DONNÉES
@st.cache_resource
def charger_modele():
    with open('modele_segmentation.pkl', 'rb') as fichier:
        return pickle.load(fichier)

@st.cache_data
def charger_donnees():
    return pd.read_csv('Resultat_Segmentation.csv')

modele = charger_modele()
df_historique = charger_donnees()

# 4. STRUCTURE EN COLONNES
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("🎛️ Simulateur de Profil")
    st.info("Ajustez les paramètres pour simuler l'arrivée d'un nouveau prospect en temps réel.")
    
    revenu = st.slider("💰 Revenu Annuel estimé (k$)", min_value=10, max_value=150, value=60, step=1)
    score = st.slider("🔥 Score d'engagement (1-100)", min_value=1, max_value=100, value=50, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyser = st.button("🚀 LANCER L'ANALYSE PRÉDICTIVE")

with col2:
    # --- LES 3 ONGLETS ---
    tab_graphique, tab_explications, tab_contexte = st.tabs(["📊 Diagnostic en direct", "📖 Dictionnaire des Profils", "🧠 Méthodologie & Contexte"])
    
    # ONGLET 1 : GRAPHIQUE ET RÉSULTAT
    with tab_graphique:
        couleurs_clusters = {1: '#3498db', 2: '#f1c40f', 3: '#e74c3c', 4: '#9b59b6', 5: '#2ecc71'}
        df_historique['Couleur'] = df_historique['Cluster'].map(couleurs_clusters)
        
        fig = px.scatter(df_historique, x='Annual Income (k$)', y='Spending Score (1-100)', 
                         color='Cluster', title="Cartographie spatiale de la base client",
                         template="plotly_dark", hover_data=['Age'])
        fig.update_traces(marker=dict(size=10, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
        
        if analyser:
            donnees_prospect = pd.DataFrame({'Annual Income (k$)': [revenu], 'Spending Score (1-100)': [score]})
            prediction = modele.predict(donnees_prospect)[0]
            
            # Animation visuelle sur le graphique
            fig.add_scatter(x=[revenu], y=[score], mode='markers+text', 
                            marker=dict(color='white', size=25, symbol='star', line=dict(color='red', width=2)),
                            name="PROSPECT SIMULÉ", text=["VOUS ÊTES ICI"], textposition="top center")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Résultat de la classification algorithmique")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Revenu Saisi", f"{revenu} k$")
            kpi2.metric("Score Saisi", f"{score} pts")
            kpi3.metric("Segment Prédit", f"Cluster {prediction}")
            
            st.markdown("### Plan d'action recommandé immédiat :")
            if prediction == 2:
                st.success("**💎 VIP (Haut potentiel) :** Ce client possède un fort pouvoir d'achat et une propension à dépenser confirmée. Déployez le service de conciergerie premium. Évitez les remises de prix qui dévalorisent l'offre, misez sur l'exclusivité et la qualité de service.")
                st.balloons()
            elif prediction == 4:
                st.warning("**⚠️ Potentiel Inexploité :** Risque de churn ou de non-conversion. Le budget est présent, mais l'offre actuelle ne déclenche pas l'achat. Action : Stratégie de réassurance, appels sortants ciblés pour comprendre les freins, garantie satisfait ou remboursé.")
            elif prediction == 3:
                st.error("**🔥 Les Insouciants :** Population très volatile. Forte sensibilité aux tendances et à la gratification immédiate. Action : Push marketing direct (SMS, notifications), offres flash à durée limitée. Attention à surveiller les impayés potentiels.")
            elif prediction == 1:
                st.info("**📊 Classe Moyenne :** Le cœur du réacteur économique. Comportement rationnel et stable. Action : Maintenir l'engagement par des newsletters régulières, intégration au programme de fidélité standard. Coût d'acquisition à maintenir bas.")
            else:
                st.markdown("<div style='padding: 1rem; border-radius: 0.5rem; background-color: #2e3b4e; color: white;'><b>💰 Les Économes :</b> Clients hyper-rationnels guidés uniquement par le prix. Action : Ne pas investir de budget d'acquisition lourd (Google Ads coûteux). Proposez le service de base strict ou les offres de déstockage/heures creuses.</div>", unsafe_allow_html=True)
                
        else:
            st.plotly_chart(fig, use_container_width=True)

    # ONGLET 2 : DICTIONNAIRE
    with tab_explications:
        st.header("L'Anatomie de nos Segments")
        st.write("Ce dictionnaire permet aux équipes opérationnelles (Marketing, Ventes, Support) d'adapter leur discours en fonction du segment identifié par l'algorithme.")
        
        st.markdown("""
        * **💎 Cluster 2 (VIP) :** La cible la plus rentable. Ils attendent de la reconnaissance, du confort et de la rapidité.
        * **📊 Cluster 1 (Classe Moyenne) :** Ils cherchent le meilleur rapport qualité/prix. Ils sont fidèles si le service est constant et sans mauvaise surprise.
        * **🔥 Cluster 3 (Insouciants) :** Ils achètent sur un coup de tête. L'expérience d'achat doit être extrêmement fluide et immédiate (paiement en 1 clic).
        * **⚠️ Cluster 4 (Potentiels) :** Ils sont exigeants ou méfiants. Il faut faire preuve d'autorité et de preuve sociale (avis clients, certifications) pour les rassurer.
        * **💰 Cluster 5 (Économes) :** Ils calculent tout. Le seul argument valable est l'économie réalisée par rapport aux concurrents.
        """)

    # ONGLET 3 : LE CONTEXTE BUSINESS (Nouveau !)
    with tab_contexte:
        st.header("Pourquoi cette plateforme a-t-elle été créée ?")
        st.write("Dans un marché ultra-concurrentiel, appliquer la même stratégie marketing à tous les clients est une perte d'argent. Cette plateforme vise à automatiser la personnalisation.")
        
        st.subheader("1. L'Algorithme utilisé (Sous le capot)")
        st.write("Nous avons utilisé un modèle d'Apprentissage Non Supervisé (**K-Means**). L'algorithme a analysé des milliers de combinaisons mathématiques pour trouver la structure naturelle de notre base de données. Il a isolé mathématiquement **5 groupes homogènes** avec un niveau de précision optimal (validé par la méthode de la variance intra-classe 'Elbow Method').")
        
        st.subheader("2. Le Modèle Prédictif")
        st.write("Pour le temps réel, nous avons superposé un **Arbre de Décision (Decision Tree Classifier)** au-dessus du K-Means. Cela permet au système, en moins de 0.01 seconde, de scanner un nouveau profil entrant et de l'assigner au bon groupe avec une précision mesurée à plus de 95%.")
        
        st.subheader("3. Le Retour sur Investissement (ROI) attendu")
        st.markdown("""
        * **Baisse du Coût d'Acquisition (CAC) :** En arrêtant de cibler les 'Économes' avec des publicités premium très chères.
        * **Hausse
