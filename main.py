# 1. IMPORTATION DES BIBLIOTHÈQUES
import pandas as pd  # Pour manipuler le tableau (Dataframe)
import numpy as np   # Pour les calculs mathématiques
import matplotlib.pyplot as plt # Pour les graphiques de base
import seaborn as sns # Pour les graphiques "stylés"
from sklearn.cluster import KMeans # L'algorithme de segmentation

# 2. CHARGEMENT DES DONNÉES
# Assure-toi que le fichier est bien à côté de ton script python
df = pd.read_csv('Mall_Customers.csv')

# On jette un oeil rapide pour voir si tout va bien
print("--- APERÇU DES DONNÉES ---")
print(df.head()) # Affiche les 5 premières lignes

# 3. EXPLORATION
print("\n--- INFOS TECHNIQUES ---")
print(df.info()) # Vérifie s'il y a des valeurs manquantes (Null)

print("\n--- STATISTIQUES ---")
print(df.describe()) # Moyenne, écart-type, min, max...

# 4. SÉLECTION DES VARIABLES PERTINENTES
# On ne garde que les colonnes 3 et 4 (Annual Income et Spending Score)
# Note : En Python, on compte à partir de 0. Donc [:, [3, 4]] prend les colonnes index 3 et 4.
X = df.iloc[:, [3, 4]].values

print("\n--- DONNÉES PRÊTES POUR LE CLUSTERING ---")
print(X[:5]) # On affiche les 5 premières lignes de notre matrice X

# 5. RECHERCHE DU NOMBRE OPTIMAL DE CLUSTERS (ELBOW METHOD)
wcss = [] # On crée une liste vide pour stocker les résultats (Inertie)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X) # L'algo s'entraîne sur nos données X
    wcss.append(kmeans.inertia_) # On sauvegarde le score d'inertie

# On affiche le graphique pour décider
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o', color='red')
plt.title('La Méthode du Coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS (Inertie)')
plt.show() # C'est important pour voir la fenêtre s'ouvrir !

# 6. CRÉATION DU MODÈLE FINAL AVEC K=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# L'algo apprend ET nous donne le groupe de chaque client d'un coup
y_kmeans = kmeans.fit_predict(X)

# On affiche le groupe assigné aux 5 premiers clients
print("\n--- GROUPES ASSIGNÉS (0 à 4) ---")
print(y_kmeans[:5])

# 7. VISUALISATION DES CLUSTERS
plt.figure(figsize=(10,7))

# On dessine chaque groupe un par un avec une couleur différente
# X[y_kmeans == 0, 0] veut dire : Dans X, prends les lignes du groupe 0, et la colonne 0 (Revenu)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1 (Radins ?)')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2 (Moyens)')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3 (Cibles VIP)')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4 (Insouciants)')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5 (Économes)')

# On ajoute les CENTRES (Centroïdes) en jaune
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', edgecolors='black')

plt.title('Segmentation des Clients')
plt.xlabel('Revenu Annuel (k$)')
plt.ylabel('Score de Dépense (1-100)')
plt.legend()
plt.show()

# 8. ANALYSE ET EXPORT
# On remet le résultat dans le tableau original pour savoir qui est qui
df['Cluster'] = y_kmeans + 1 # On fait +1 pour avoir des groupes de 1 à 5 (plus joli que 0 à 4)

# On calcule la moyenne de chaque groupe pour comprendre qui ils sont
print("\n--- PROFIL MOYEN DES GROUPES ---")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

# Optionnel : Sauvegarder le résultat dans un nouveau fichier Excel/CSV
df.to_csv('Resultat_Segmentation.csv', index=False)
print("\nFichier 'Resultat_Segmentation.csv' généré avec succès !")