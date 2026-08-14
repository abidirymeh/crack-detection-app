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

data = {
    'age_batiment': np.random.randint(1, 100, n_echantillons),  
    'humidite': np.random.uniform(30, 90, n_echantillons),     
    'temperature': np.random.uniform(-5, 40, n_echantillons),   
    'indice_usure': np.random.uniform(0, 10, n_echantillons),   
    'nb_fissures': np.random.randint(0, 50, n_echantillons),    
    'resistance': np.random.uniform(10, 50, n_echantillons)     
}

df = pd.DataFrame(data)

print(f"\nDataset créé: {len(df)} échantillons, {len(df.columns)} colonnes")
print(f"\n Aperçu des données:")
print(df.head())

print(f"\n Statistiques descriptives:")
print(df.describe())

print("\n  Génération des graphiques...")

plt.figure(figsize=(15, 10))

for i, col in enumerate(df.columns[:-1]):  # sauf 'resistance'
    plt.subplot(2, 3, i+1)
    plt.hist(df[col], bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution - {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.savefig('distribution_materiaux.png')
print(" Distribution sauvegardée: distribution_materiaux.png")

plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation - Données matériaux')
plt.tight_layout()
plt.savefig('correlation_materiaux.png')
print(" Matrice de corrélation sauvegardée: correlation_materiaux.png")

print("\n ÉTAPE 7.1 TERMINÉE!")