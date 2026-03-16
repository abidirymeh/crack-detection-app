# step7_4_materiaux_tester.py
import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("=" * 70)
print("ÉTAPE 7.4: TEST DU MODÈLE MATÉRIAUX")
print("=" * 70)

# 1. CHARGER LE MODÈLE ET LE SCALER
print("\n📥 Chargement du modèle...")
model = tf.keras.models.load_model('modele_materiaux_final.h5')
scaler = joblib.load('scaler_materiaux.pkl')
print(" Modèle et scaler chargés")

# 2. TEST SUR QUELQUES EXEMPLES
print("\n🔍 TEST SUR EXEMPLES SIMULÉS")
print("-" * 50)

exemples = [
    [10, 45, 20, 2, 5],    # Bâtiment récent, peu d'usure
    [50, 70, 15, 7, 25],    # Bâtiment ancien, très usé
    [80, 85, 10, 9, 40],    # Très vieux, très humide
    [5, 30, 25, 1, 2],      # Neuf, bon état
]

for i, ex in enumerate(exemples):
    # Normaliser
    ex_scaled = scaler.transform([ex])
    
    # Prédire
    pred = model.predict(ex_scaled, verbose=0)[0][0]
    
    print(f"\n Cas {i+1}:")
    print(f"   Age: {ex[0]} ans, Humidité: {ex[1]}%, Temp: {ex[2]}°C")
    print(f"   Usure: {ex[3]}/10, Fissures: {ex[4]}")
    print(f"   → Résistance estimée: {pred:.1f} MPa")
    
    if pred < 20:
        print("     DANGER: Résistance très faible!")
    elif pred < 30:
        print("     Attention: Résistance modérée")
    else:
        print("    OK: Résistance bonne")

print("\n ÉTAPE 7.4 TERMINÉE!")