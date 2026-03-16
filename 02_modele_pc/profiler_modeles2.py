import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import csv
from datetime import datetime

print("=" * 70)
print("PROFILING DES MODELES LIGHTWEIGHT POUR DETECTION DE FISSURES")
print("=" * 70)

# Configuration
IMG_SIZE = 224
NUM_IMAGES_TEST = 100  # Nombre d'images pour les tests

# Liste des modèles à tester
modeles = [
    {
        "nom": "MobileNetV3-Small",
        "constructeur": lambda: tf.keras.applications.MobileNetV3Small(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            classes=1000  # ImageNet a 1000 classes
        )
    },
    {
        "nom": "MobileNetV2",
        "constructeur": lambda: tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            classes=1000
        )
    },
    {
        "nom": "EfficientNet-B0",
        "constructeur": lambda: tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            classes=1000
        )
    },
    {
        "nom": "ShuffleNetV2",
        "constructeur": lambda: tf.keras.applications.ShuffleNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            classes=1000
        )
    }
]

# Créer des données factices pour les tests (en attendant SDNET2018)
def generer_donnees_test(nb_images=100):
    X_test = np.random.rand(nb_images, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, nb_images)  # 0: pas fissure, 1: fissure
    return X_test, y_test

print("\n🔬 Génération des données de test...")
X_test, y_test = generer_donnees_test(NUM_IMAGES_TEST)
print(f"✅ {NUM_IMAGES_TEST} images de test générées")

# Tableau des résultats
resultats = []
fichier_resultats = f"resultats_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

for modele_info in modeles:
    print(f"\n{'='*50}")
    print(f"📊 ANALYSE DU MODÈLE : {modele_info['nom']}")
    print(f"{'='*50}")
    
    # 1. Charger le modèle
    print("⏳ Chargement du modèle...")
    modele = modele_info["constructeur"]()
    print("✅ Modèle chargé")
    
    # 2. Mesurer le nombre de paramètres
    params = modele.count_params()
    print(f"📈 Paramètres: {params:,} ({params/1e6:.2f} millions)")
    
    # 3. Mesurer la taille du fichier (sauvegarde temporaire)
    print("⏳ Calcul de la taille...")
    fichier_temp = f"temp_{modele_info['nom']}.h5"
    modele.save(fichier_temp)
    taille_fichier = os.path.getsize(fichier_temp) / (1024 * 1024)  # en Mo
    os.remove(fichier_temp)
    print(f"💾 Taille du modèle: {taille_fichier:.2f} Mo")
    
    # 4. Calculer les FLOPS (si keras-flops est installé)
    try:
        from keras_flops import get_flops
        flops = get_flops(modele, batch_size=1)
        print(f"⚡ FLOPS: {flops/1e9:.2f} G")
    except:
        flops = "N/A"
        print("⚡ FLOPS: Non calculable (keras-flops non disponible)")
    
    # 5. Mesurer la latence sur CPU
    print("⏳ Mesure de la latence...")
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    # Warm-up (pour éviter les mesures faussées)
    for _ in range(10):
        modele.predict(dummy_input, verbose=0)
    
    # Mesure réelle sur 50 itérations
    latences = []
    for i in range(50):
        start = time.time()
        modele.predict(dummy_input, verbose=0)
        end = time.time()
        latences.append((end - start) * 1000)  # en ms
    
    latence_moyenne = np.mean(latences)
    latence_std = np.std(latences)
    print(f"⏱️  Latence moyenne: {latence_moyenne:.2f} ms (±{latence_std:.2f})")
    
    # 6. Simuler des prédictions pour les métriques
    print("⏳ Calcul des métriques de classification...")
    # Prendre un sous-ensemble pour les métriques
    y_pred_proba = modele.predict(X_test[:50], verbose=0)
    
    # Pour la binarisation, on prend la classe avec la plus haute proba
    y_pred = np.argmax(y_pred_proba, axis=1)
    # On convertit en binaire (fissure/pas fissure) - simplification
    y_pred_binary = (y_pred > 500).astype(int)  # Seuil arbitraire car ImageNet a 1000 classes
    
    # 7. Calculer les métriques de classification
    accuracy = accuracy_score(y_test[:50], y_pred_binary)
    precision = precision_score(y_test[:50], y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_test[:50], y_pred_binary, average='binary', zero_division=0)
    f1 = f1_score(y_test[:50], y_pred_binary, average='binary', zero_division=0)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Précision: {precision:.4f}")
    print(f"🎯 Rappel: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}  <--- MÉTRIQUE CLÉ")
    
    # Sauvegarder les résultats
    resultats.append({
        "Modèle": modele_info["nom"],
        "Paramètres (M)": f"{params/1e6:.2f}",
        "Taille (Mo)": f"{taille_fichier:.2f}",
        "FLOPS (G)": f"{flops/1e9:.2f}" if flops != "N/A" else "N/A",
        "Latence (ms)": f"{latence_moyenne:.2f}",
        "Latence_std (ms)": f"{latence_std:.2f}",
        "Accuracy": f"{accuracy:.4f}",
        "Précision": f"{precision:.4f}",
        "Rappel": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}"
    })
    
    print(f"✅ Analyse de {modele_info['nom']} terminée")

# Afficher le tableau récapitulatif
print("\n" + "=" * 100)
print("📋 TABLEAU RÉCAPITULATIF DES RÉSULTATS")
print("=" * 100)

for r in resultats:
    print(f"{r['Modèle']:20} | Taille: {r['Taille (Mo)']:>5} Mo | Latence: {r['Latence (ms)']:>6} ms | F1: {r['F1-Score']}")

# Sauvegarder dans un fichier CSV
with open(fichier_resultats, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)

print(f"\n💾 Résultats sauvegardés dans : {fichier_resultats}")
print("\n✅ Phase de profiling sur PC terminée !")
print("📁 Tu peux maintenant ouvrir ce fichier CSV avec Excel pour voir les résultats détaillés.")