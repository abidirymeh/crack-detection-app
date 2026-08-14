import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 70)
print("ÉTAPE 7.2: PRÉPARATION DES DONNÉES MATÉRIAUX")
print("=" * 70)

print("\n Génération des données...")

np.random.seed(42)
n_echantillons = 1000

age = np.random.randint(1, 100, n_echantillons)
humidite = np.random.uniform(30, 90, n_echantillons)
temperature = np.random.uniform(-5, 40, n_echantillons)
usure = np.random.uniform(0, 10, n_echantillons)
nb_fissures = np.random.randint(0, 50, n_echantillons)

resistance = (50 - 0.3*age - 0.1*humidite - 0.05*temperature - 1.5*usure - 0.2*nb_fissures 
              + np.random.normal(0, 3, n_echantillons))
resistance = np.clip(resistance, 5, 60)  # Garder dans des limites réalistes

data = {
    'age_batiment': age,
    'humidite': humidite,
    'temperature': temperature,
    'indice_usure': usure,
    'nb_fissures': nb_fissures,
    'resistance': resistance
}

df = pd.DataFrame(data)
print(f" Dataset créé: {len(df)} échantillons")

X = df[['age_batiment', 'humidite', 'temperature', 'indice_usure', 'nb_fissures']]
y = df['resistance']

print(f"\n Features: {list(X.columns)}")
print(f" Target: resistance (MPa)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Split:")
print(f"   - Train: {len(X_train)} échantillons")
print(f"   - Test: {len(X_test)} échantillons")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n Statistiques après normalisation:")
print(f"   - Moyenne train: {X_train_scaled.mean():.2f}")
print(f"   - Écart-type train: {X_train_scaled.std():.2f}")

np.savez('materiaux_prepared.npz',
         X_train=X_train_scaled, X_test=X_test_scaled,
         y_train=y_train.values, y_test=y_test.values)

joblib.dump(scaler, 'scaler_materiaux.pkl')
print(f"\n Données sauvegardées: materiaux_prepared.npz")
print(f" Scaler sauvegardé: scaler_materiaux.pkl")

print("\n ÉTAPE 7.2 TERMINÉE!")