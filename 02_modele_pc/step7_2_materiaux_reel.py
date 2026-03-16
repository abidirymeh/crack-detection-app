# step7_2_materiaux_reel.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("CHARGEMENT DU DATASET UCI CONCRETE RÉEL")
print("=" * 70)

# 1. CHARGER LE FICHIER
print("\n📥 Chargement du fichier...")

# Essayer différents formats

    # Si c'est un fichier Excel
df = pd.read_excel('C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/Concrete_Data.xls', header=None)
print("✅ Fichier Excel chargé")

print(f"\n📊 Aperçu des données:")
print(df.head())

print(f"\n📊 Informations:")
print(df.info())

# 2. AJOUTER LES NOMS DE COLONNES (pour UCI Concrete)
colonnes = [
    'Ciment', 'Laitier', 'Cendres_volantes', 'Eau',
    'Superplastifiant', 'Granulat_gros', 'Granulat_fin', 'Age',
    'Resistance'
]

# Si le fichier n'a pas d'en-tête, ajouter les noms
# Si le fichier n'a pas d'en-tête, ajouter les noms
if df.shape[1] == 9:  # UCI Concrete a 9 colonnes
    df.columns = colonnes

    # 🔥 NOUVELLES LIGNES 🔥
    print("\n🔄 Conversion des données en nombres...")
    # La première ligne contient les descriptions, on la supprime
    df = df.iloc[1:].copy()
    
    # Convertir toutes les colonnes en nombres
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    
    print(f"✅ Conversion terminée: {len(df)} échantillons valides")
    print(f"\n📊 Types après conversion:")
    print(df.dtypes)
print(f"\n📋 Colonnes:")
for i, col in enumerate(df.columns):
    print(f"   {i}: {col}")

# 3. STATISTIQUES DESCRIPTIVES
print("\n📈 Statistiques:")
print(df.describe())

# 4. VISUALISATION DES DISTRIBUTIONS
print("\n📊 Génération des graphiques...")

plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:-1]):  # sauf Resistance
    plt.subplot(3, 3, i+1)
    plt.hist(df[col], bins=30, alpha=0.7)
    plt.title(col)
plt.tight_layout()
plt.savefig('distribution_concrete.png')
print("✅ Distribution sauvegardée: distribution_concrete.png")

# 5. MATRICE DE CORRÉLATION
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Corrélations - UCI Concrete Dataset')
plt.tight_layout()
plt.savefig('correlation_concrete.png')
print("✅ Corrélations sauvegardées: correlation_concrete.png")

# 6. SÉPARATION FEATURES/TARGET
X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
y = df.iloc[:, -1]   # Dernière colonne = Resistance

print(f"\n🎯 Features: {list(X.columns)}")
print(f"🎯 Target: {df.columns[-1]}")

# 7. DIVISION TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Split:")
print(f"   - Train: {len(X_train)} échantillons")
print(f"   - Test: {len(X_test)} échantillons")

# 8. NORMALISATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. SAUVEGARDER
np.savez('concrete_prepared.npz',
         X_train=X_train_scaled, X_test=X_test_scaled,
         y_train=y_train.values, y_test=y_test.values)

joblib.dump(scaler, 'scaler_concrete.pkl')
print(f"\n💾 Données sauvegardées: concrete_prepared.npz")
print(f"💾 Scaler sauvegardé: scaler_concrete.pkl")

print("\n✅ DATASET PRÊT POUR L'ENTRAÎNEMENT!")