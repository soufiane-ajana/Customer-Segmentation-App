# 1. IMPORTATION 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 2. CHARGEMENT DES DONNÉES SEGMENTÉES
# On utilise le fichier qu'on a créé à l'étape précédente
df = pd.read_csv('Resultat_Segmentation.csv')

# 3. SÉPARATION DES VARIABLES (X = Les indices, y = La réponse à trouver)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']] # Ce que l'IA regarde
y = df['Cluster'] # Ce que l'IA doit deviner

# 4. SÉPARATION EN DONNÉES D'ENTRAÎNEMENT ET DE TEST
# On cache 20% des clients à l'IA pour vérifier si elle est intelligente après coup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entraînement sur", len(X_train), "clients.")
print("Test sur", len(X_test), "clients.")

# 5. CRÉATION ET ENTRAÎNEMENT DU MODÈLE
# On limite la profondeur de l'arbre à 4 pour qu'il reste facile à comprendre
modele_arbre = DecisionTreeClassifier(max_depth=4, random_state=42)

# L'IA apprend les règles métier ici :
modele_arbre.fit(X_train, y_train)

# 6. ÉVALUATION (Le modèle est-il bon ?)
precision = modele_arbre.score(X_test, y_test)
print(f"\nPrécision de l'IA sur les nouveaux clients : {precision * 100}%")

# 7. PRÉDICTION SUR DE NOUVEAUX CLIENTS (En Temps Réel)
# Imaginons 2 nouveaux clients qui s'inscrivent :
# Client A : Gagne 85k$ par an, mais a un score de dépense très faible (12)
# Client B : Gagne 20k$ par an, mais a un score de dépense explosif (90)

nouveaux_clients = pd.DataFrame({
    'Annual Income (k$)': [85, 20],
    'Spending Score (1-100)': [12, 90]
})

predictions = modele_arbre.predict(nouveaux_clients)

print("\n--- RÉSULTATS DES PRÉDICTIONS EN TEMPS RÉEL ---")
print(f"Le Client A a été classé automatiquement dans le Cluster : {predictions[0]}")
print(f"Le Client B a été classé automatiquement dans le Cluster : {predictions[1]}")

# 8. VISUALISATION DES RÈGLES DE DÉCISION
plt.figure(figsize=(15, 10))
tree.plot_tree(modele_arbre, 
               feature_names=['Revenu (k$)', 'Score Dépense'],  
               class_names=['1', '2', '3', '4', '5'],
               filled=True, 
               rounded=True, 
               fontsize=10)

plt.title("Le 'Cerveau' de l'IA : Règles de classification automatique")
plt.show()

# 9. SAUVEGARDE DU MODÈLE (Déploiement)
import pickle # La bibliothèque Python pour "congeler" des objets

# On crée un fichier '.pkl' en mode écriture binaire ('wb' = write binary)
nom_du_fichier = 'modele_segmentation.pkl'

with open(nom_du_fichier, 'wb') as fichier:
    pickle.dump(modele_arbre, fichier) # On verse le "cerveau" dans le fichier

print(f"\nSuccès ! L'IA a été sauvegardée sous le nom : {nom_du_fichier}")