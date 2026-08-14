# step7_5_integration.py (version finale)
import tensorflow as tf
import numpy as np
import joblib
import cv2
import os

print("=" * 70)
print("ÉTAPE 7.5: INTÉGRATION PARTIES 1 & 2")
print("=" * 70)

# Charger les deux modèles
print("\n Chargement des modèles...")

interpreter_fissures = tf.lite.Interpreter(model_path='../detection_fissures/modele_enriched.tflite')
interpreter_fissures.allocate_tensors()

model_materiaux = tf.keras.models.load_model('modele_materiaux_final.h5')
scaler = joblib.load('scaler_materiaux.pkl')

print(" Modèles chargés")

def analyser_batiment(image_path, donnees_capteurs):
    # Vérifier image
    if not os.path.exists(image_path):
        print(f"\n❌ Image non trouvée: {image_path}")
        print("   Utilisation d'une image par défaut...")
        return None, None
    
    # Analyse visuelle
    print(f"\n Analyse de l'image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("   ❌ Erreur chargement")
        return None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    
    interpreter_fissures.set_tensor(
        interpreter_fissures.get_input_details()[0]['index'], 
        img_input
    )
    interpreter_fissures.invoke()
    proba_fissure = interpreter_fissures.get_tensor(
        interpreter_fissures.get_output_details()[0]['index']
    )[0][0]
    
    SEUIL = 0.46
    print(f"   → Probabilité fissure: {proba_fissure:.2%}")
    print(f"   → {'🔴 FISSURE' if proba_fissure > SEUIL else '🟢 SAIN'}")
    
    # Analyse matériaux
    print(f"\n Analyse des matériaux:")
    print(f"   Age: {donnees_capteurs[0]} ans")
    print(f"   Humidité: {donnees_capteurs[1]}%")
    print(f"   Température: {donnees_capteurs[2]}°C")
    print(f"   Usure: {donnees_capteurs[3]}/10")
    print(f"   Nb fissures: {donnees_capteurs[4]}")
    
    capteurs_scaled = scaler.transform([donnees_capteurs])
    resistance = model_materiaux.predict(capteurs_scaled, verbose=0)[0][0]
    
    print(f"   → Résistance: {resistance:.1f} MPa")
    
    if resistance < 20:
        print("   ⚠️ DANGER: Résistance très faible!")
    elif resistance < 30:
        print("   ⚠️ Attention: Résistance modérée")
    else:
        print("   ✅ OK: Résistance bonne")
    
    # Recommandation
    print("\n🏁 RECOMMANDATION:")
    if proba_fissure > SEUIL and resistance < 25:
        print("   🚨 URGENT: Inspection immédiate!")
    elif proba_fissure > SEUIL or resistance < 25:
        print("   ⚠️ Surveillance renforcée")
    else:
        print("   ✅ Bâtiment en bon état")
    
    return proba_fissure, resistance

# TEST
if __name__ == "__main__":
    print("\n🔍 TEST D'INTÉGRATION")
    print("=" * 50)
    
    donnees_test = [25, 65, 18, 4, 12]
    
    # METTRE UN CHEMIN QUI EXISTE VRAIMENT
    image_test = r"C:\Users\RIMEH\Desktop\PFA_ADEVA\03_dataset\testImage\fissure\1.jpeg"
    
    analyser_batiment(image_test, donnees_test)

print("\n✅ ÉTAPE 7.5 TERMINÉE!")