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

print("\n Chargement du fichier...")


df = pd.read_excel('C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/Concrete_Data.xls', header=None)
print(" Fichier Excel chargé")

print(f"\n Aperçu des données:")
print(df.head())

print(f"\n Informations:")
print(df.info())

colonnes = [
    'Ciment', 'Laitier', 'Cendres_volantes', 'Eau',
    'Superplastifiant', 'Granulat_gros', 'Granulat_fin', 'Age',
    'Resistance'
]


if df.shape[1] == 9:  
    df.columns = colonnes

    print("\n🔄 Conversion des données en nombres...")
    df = df.iloc[1:].copy()
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    print(f" Conversion terminée: {len(df)} échantillons valides")
    print(f"\n Types après conversion:")
    print(df.dtypes)
print(f"\n Colonnes:")
for i, col in enumerate(df.columns):
    print(f"   {i}: {col}")

print("\n Statistiques:")
print(df.describe())

print("\n Génération des graphiques...")

plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:-1]):  # sauf Resistance
    plt.subplot(3, 3, i+1)
    plt.hist(df[col], bins=30, alpha=0.7)
    plt.title(col)
plt.tight_layout()
plt.savefig('distribution_concrete.png')
print(" Distribution sauvegardée: distribution_concrete.png")

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Corrélations - UCI Concrete Dataset')
plt.tight_layout()
plt.savefig('correlation_concrete.png')
print(" Corrélations sauvegardées: correlation_concrete.png")

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   

print(f"\n Features: {list(X.columns)}")
print(f" Target: {df.columns[-1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Split:")
print(f"   - Train: {len(X_train)} échantillons")
print(f"   - Test: {len(X_test)} échantillons")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.savez('concrete_prepared.npz',
         X_train=X_train_scaled, X_test=X_test_scaled,
         y_train=y_train.values, y_test=y_test.values)

joblib.dump(scaler, 'scaler_concrete.pkl')
print(f"\n Données sauvegardées: concrete_prepared.npz")
print(f" Scaler sauvegardé: scaler_concrete.pkl")

print("\n DATASET PRÊT POUR L'ENTRAÎNEMENT!")