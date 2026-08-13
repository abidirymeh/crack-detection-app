# step7_5_integration.py
import tensorflow as tf
import numpy as np
import joblib
import cv2
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 70)
print("ÉTAPE 7.5: INTÉGRATION PARTIES 1 & 2")
print("=" * 70)

# Charger les deux modèles
print("\n Chargement des modèles...")

# Modèle de détection de fissures (Partie 1)
interpreter_fissures = tf.lite.Interpreter(model_path='mobilenetv3_fissures.tflite')
interpreter_fissures.allocate_tensors()

# Modèle de prédiction matériaux (Partie 2)
model_materiaux = tf.keras.models.load_model('modele_materiaux_final.h5')
scaler = joblib.load('scaler_materiaux.pkl')

print(" Modèles chargés")

def analyser_batiment(image_path, donnees_capteurs):
    """
    Analyse complète d'un bâtiment:
    - Partie 1: Détection de fissures sur photo
    - Partie 2: Prédiction de durabilité avec données capteurs
    """
    
    # PARTIE 1: ANALYSE VISUELLE
    print(f"\n Analyse de l'image: {image_path}")
    
    # Charger et préparer l'image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    
    # Prédire
    interpreter_fissures.set_tensor(
        interpreter_fissures.get_input_details()[0]['index'], 
        img_input
    )
    interpreter_fissures.invoke()
    proba_fissure = interpreter_fissures.get_tensor(
        interpreter_fissures.get_output_details()[0]['index']
    )[0][0]
    
    print(f"   → Probabilité fissure: {proba_fissure:.2%}")
    print(f"   → {' FISSURE DÉTECTÉE' if proba_fissure > 0.5 else ' PAS DE FISSURE'}")
    
    # PARTIE 2: ANALYSE MATÉRIAUX
    print(f"\n Analyse des matériaux:")
    print(f"   Age: {donnees_capteurs[0]} ans")
    print(f"   Humidité: {donnees_capteurs[1]}%")
    print(f"   Température: {donnees_capteurs[2]}°C")
    print(f"   Indice d'usure: {donnees_capteurs[3]}/10")
    print(f"   Nombre de fissures: {donnees_capteurs[4]}")
    
    # Normaliser et prédire
    capteurs_scaled = scaler.transform([donnees_capteurs])
    resistance = model_materiaux.predict(capteurs_scaled, verbose=0)[0][0]
    
    print(f"   → Résistance estimée: {resistance:.1f} MPa")
    
    if resistance < 20:
        print("   ⚠️  DANGER: Résistance très faible!")
    elif resistance < 30:
        print("   ⚠️  Attention: Résistance modérée")
    else:
        print("   ✅ OK: Résistance bonne")
    
    # RECOMMANDATION FINALE
    print("\n🏁 RECOMMANDATION:")
    if proba_fissure > 0.5 and resistance < 25:
        print("   🚨 URGENT: Inspection requise immédiatement!")
    elif proba_fissure > 0.5 or resistance < 25:
        print("   ⚠️  Surveillance renforcée recommandée")
    else:
        print("   ✅ Bâtiment en bon état, surveillance normale")
    
    return proba_fissure, resistance

# TEST
if __name__ == "__main__":
    # Exemple de test
    print("\n🔍 TEST D'INTÉGRATION")
    print("=" * 50)
    
    # Données capteurs simulées
    donnees_test = [25, 65, 18, 4, 12]  # Age, humidité, temp, usure, nb fissures
    
    # Chemin vers une image de test (à adapter)
    image_test = "test_fissure.jpg"  # Mets le chemin de ton image
    
    analyser_batiment(image_test, donnees_test)

print("\n✅ ÉTAPE 7.5 TERMINÉE!")