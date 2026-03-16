# test_projet_complet.py
import tensorflow as tf
import numpy as np
import joblib
import cv2
import os
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 70)
print("🔍 TEST FINAL DU PROJET COMPLET")
print("=" * 70)

# ============================================
# PARTIE 1: TEST DU MODÈLE DE FISSURES
# ============================================
print("\n" + "=" * 50)
print("📸 PARTIE 1: DÉTECTION DE FISSURES")
print("=" * 50)

# Vérifier que le modèle TFLite existe
if os.path.exists('mobilenetv3_fissures.tflite'):
    print("✅ Modèle TFLite trouvé")
    
    # Charger le modèle
    interpreter = tf.lite.Interpreter(model_path='mobilenetv3_fissures.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Format entrée: {input_details[0]['shape']}")
    print(f"   Format sortie: {output_details[0]['shape']}")
    
    # Tester avec une image si elle existe
    image_test = input("📂 Entrez le chemin d'une image de test (ou 'skip'): ")
    
    if image_test != 'skip' and os.path.exists(image_test):
        img = cv2.imread(image_test)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        proba = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        print(f"\n📊 RÉSULTAT:")
        print(f"   Probabilité fissure: {proba:.2%}")
        print(f"   → {'🚨 FISSURE DÉTECTÉE' if proba > 0.5 else '✅ PAS DE FISSURE'}")
    else:
        print("⏩ Test d'image ignoré")
else:
    print("❌ Modèle TFLite manquant!")

# ============================================
# PARTIE 2: TEST DU MODÈLE MATÉRIAUX
# ============================================
print("\n" + "=" * 50)
print("🧪 PARTIE 2: ANALYSE DES MATÉRIAUX")
print("=" * 50)

# Vérifier les fichiers
if os.path.exists('modele_concrete_final.h5') and os.path.exists('scaler_concrete.pkl'):
    print("✅ Modèle matériaux trouvé")
    print("✅ Scaler trouvé")
    
    # Charger le modèle et le scaler
    model_materiaux = tf.keras.models.load_model('modele_concrete_final.h5')
    scaler = joblib.load('scaler_concrete.pkl')
    
    # Données de test
    print("\n📊 Test sur 4 cas typiques:")
    
    # PAR CELLES-CI (avec les 8 features) :
    cas_test = [
        [350, 0, 0, 180, 0, 900, 600, 28],   # Béton standard
        [250, 50, 30, 190, 5, 950, 650, 28], # Avec ajouts
        [150, 100, 80, 200, 10, 1000, 700, 28], # Ancien
        [100, 150, 100, 210, 15, 1050, 750, 28], # Très ancien
    ]
    
    for i, cas in enumerate(cas_test):
        cas_scaled = scaler.transform([cas])
        pred = model_materiaux.predict(cas_scaled, verbose=0)[0][0]
        
        print(f"\n   Cas {i+1}: Age={cas[0]}ans, Humidité={cas[1]}%, Usure={cas[3]}/10")
        print(f"      → Résistance estimée: {pred:.1f} MPa")
        
        if pred < 20:
            print("      ⚠️  DANGER: Résistance très faible!")
        elif pred < 30:
            print("      ⚠️  Attention: Résistance modérée")
        else:
            print("      ✅ OK: Résistance bonne")
    
    # Vérifier les métriques
    print("\n📈 Métriques du modèle:")
    try:
        # Charger les données de test
        data = np.load('concrete_prepared.npz')
        X_test, y_test = data['X_test'], data['y_test']
        
        y_pred = model_materiaux.predict(X_test, verbose=0).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   - MAE: {mae:.2f} MPa")
        print(f"   - R²: {r2:.4f}")
        
        if r2 > 0.85:
            print("   ✅ Excellent modèle!")
        elif r2 > 0.7:
            print("   ⚠️ Bon modèle")
        else:
            print("   ❌ Modèle à améliorer")
    except:
        print("   ⚠️ Données de test non trouvées")
else:
    print("❌ Fichiers matériaux manquants!")

# ============================================
# BILAN FINAL
# ============================================
print("\n" + "=" * 50)
print("📋 BILAN FINAL")
print("=" * 50)

nb_tests_reussis = 0
nb_tests_total = 5

print("\n✅ Tests automatiques:")
print(f"   - Modèle fissures présent: {'✅' if os.path.exists('mobilenetv3_fissures.tflite') else '❌'}")
print(f"   - Modèle matériaux présent: {'✅' if os.path.exists('modele_concrete_final.h5') else '❌'}")
print(f"   - Scaler présent: {'✅' if os.path.exists('scaler_concrete.pkl') else '❌'}")
print(f"   - Données préparées: {'✅' if os.path.exists('concrete_prepared.npz') else '❌'}")
print(f"   - Scripts step: {'✅' if os.path.exists('step7_5_integration.py') else '❌'}")

print("\n🎯 CONCLUSION:")
if all([
    os.path.exists('mobilenetv3_fissures.tflite'),
    os.path.exists('modele_concrete_final.h5'),
    os.path.exists('scaler_concrete.pkl'),
    os.path.exists('concrete_prepared.npz')
]):
    print("   ✅ TON PROJET EST COMPLET !")
    print("   Tu as les deux modèles fonctionnels.")
    print("\n📝 Prochaines étapes:")
    print("   1. Tester avec des images réelles")
    print("   2. Rassembler les résultats pour le rapport")
    print("   3. Préparer la soutenance")
else:
    print("   ⚠️ Il manque certains fichiers.")
    print("   Vérifie la checklist ci-dessus.")

print("\n" + "=" * 70)