# step7_1_materiaux_explorer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 70)
print("ÉTAPE 7.1: EXPLORATION DES DONNÉES MATÉRIAUX")
print("=" * 70)

print("\n Génération de données synthétiques...")

np.random.seed(42)
n_echantillons = 1000

# Creation d'un dataset simulé
data = {
    'age_batiment': np.random.randint(1, 100, n_echantillons),  # années
    'humidite': np.random.uniform(30, 90, n_echantillons),      # %
    'temperature': np.random.uniform(-5, 40, n_echantillons),    # °C
    'indice_usure': np.random.uniform(0, 10, n_echantillons),    # échelle 0-10
    'nb_fissures': np.random.randint(0, 50, n_echantillons),     # nombre
    'resistance': np.random.uniform(10, 50, n_echantillons)      # MPa (cible)
}

df = pd.DataFrame(data)

print(f"\nDataset créé: {len(df)} échantillons, {len(df.columns)} colonnes")
print(f"\n Aperçu des données:")
print(df.head())

print(f"\n Statistiques descriptives:")
print(df.describe())

# 2. VISUALISATION DES DISTRIBUTIONS
print("\n  Génération des graphiques...")

plt.figure(figsize=(15, 10))

# Distribution de chaque variable
for i, col in enumerate(df.columns[:-1]):  # sauf 'resistance'
    plt.subplot(2, 3, i+1)
    plt.hist(df[col], bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution - {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.savefig('distribution_materiaux.png')
print(" Distribution sauvegardée: distribution_materiaux.png")

# 3. CORRÉLATIONS
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation - Données matériaux')
plt.tight_layout()
plt.savefig('correlation_materiaux.png')
print(" Matrice de corrélation sauvegardée: correlation_materiaux.png")

print("\n ÉTAPE 7.1 TERMINÉE!")