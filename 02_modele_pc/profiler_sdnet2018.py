import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import cv2
from glob import glob
import csv
from datetime import datetime

print("=" * 70)
print("PROFILING DES MODELES SUR SDNET2018")
print("=" * 70)

# Configuration
IMG_SIZE = 224
MAX_IMAGES_PAR_CATEGORIE = 500  # Limite pour que ça aille plus vite

def charger_images_sdnet2018(chemin_base, max_images=500):
    """
    Charge les images depuis la structure SDNET2018
    """
    images = []
    labels = []
    sources = []  # Pour savoir d'où vient l'image (walls, decks, pavement)
    
    categories = ['walls', 'decks', 'pavement']
    
    print(f"\n📂 Scan des dossiers dans : {chemin_base}")
    
    for categorie in categories:
        chemin_cracked = os.path.join(chemin_base, categorie, 'cracked', '*.[jJ][pP][gG]')
        chemin_nocracked = os.path.join(chemin_base, categorie, 'nocracked', '*.[jJ][pP][gG]')
        
        # Images avec fissures
        fichiers_cracked = glob(chemin_cracked)
        print(f"   {categorie}/cracked: {len(fichiers_cracked)} images")
        
        for img_path in fichiers_cracked[:max_images//6]:  # Répartir entre les catégories
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)  # 1 = fissure
                sources.append(categorie)
            except Exception as e:
                print(f"   Erreur avec {img_path}: {e}")
        
        # Images sans fissures
        fichiers_nocracked = glob(chemin_nocracked)
        print(f"   {categorie}/nocracked: {len(fichiers_nocracked)} images")
        
        for img_path in fichiers_nocracked[:max_images//6]:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)  # 0 = pas fissure
                sources.append(categorie)
            except Exception as e:
                print(f"   Erreur avec {img_path}: {e}")
    
    return np.array(images), np.array(labels), sources

# Demander le chemin du dataset
chemin_dataset = input("📁 Entre le chemin vers SDNET2018 (ex: C:/Users/.../sdnet): ").strip()
if not chemin_dataset:
    chemin_dataset = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

print(f"\n📥 Chargement de SDNET2018 depuis : {chemin_dataset}")
X, y, sources = charger_images_sdnet2018(chemin_dataset, MAX_IMAGES_PAR_CATEGORIE)

print(f"\n✅ Total chargé : {len(X)} images")
print(f"   - Avec fissures: {np.sum(y)} images")
print(f"   - Sans fissures: {len(y) - np.sum(y)} images")
print(f"   - Répartition: walls={sources.count('walls')}, decks={sources.count('decks')}, pavement={sources.count('pavement')}")

# Normaliser les images
X = X / 255.0

# Diviser en train/test (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split :")
print(f"   - Train: {len(X_train)} images")
print(f"   - Test: {len(X_test)} images")

# Liste des modèles à tester
modeles = [
    {
        "nom": "MobileNetV3-Small",
        "constructeur": lambda: tf.keras.applications.MobileNetV3Small(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    },
    {
        "nom": "MobileNetV2",
        "constructeur": lambda: tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    },
    {
        "nom": "EfficientNet-B0",
        "constructeur": lambda: tf.keras.applications.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    },
    {
        "nom": "ShuffleNetV2",
        "constructeur": lambda: tf.keras.applications.ShuffleNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    }
]

# Résultats
resultats = []
fichier_resultats = f"resultats_sdnet2018_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

for modele_info in modeles:
    print(f"\n{'='*60}")
    print(f"📊 ANALYSE DU MODÈLE : {modele_info['nom']}")
    print(f"{'='*60}")
    
    # 1. Charger le modèle de base
    print("⏳ Chargement du modèle...")
    base_model = modele_info["constructeur"]()
    base_model.trainable = False  # On ne fine-tune pas pour l'instant
    
    # 2. Ajouter une tête de classification
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    
    print("✅ Modèle préparé")
    
    # 3. Compiler
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Mesurer le nombre de paramètres
    params = model.count_params()
    print(f"📈 Paramètres: {params:,} ({params/1e6:.2f} millions)")
    
    # 5. Mesurer la taille du fichier
    fichier_temp = f"temp_{modele_info['nom']}.h5"
    model.save(fichier_temp)
    taille_fichier = os.path.getsize(fichier_temp) / (1024 * 1024)
    os.remove(fichier_temp)
    print(f"💾 Taille du modèle: {taille_fichier:.2f} Mo")
    
    # 6. Mesurer la latence
    print("⏳ Mesure de la latence...")
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        model.predict(dummy_input, verbose=0)
    
    # Mesure réelle
    latences = []
    for _ in range(50):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        end = time.time()
        latences.append((end - start) * 1000)
    
    latence_moyenne = np.mean(latences)
    latence_std = np.std(latences)
    print(f"⏱️  Latence moyenne: {latence_moyenne:.2f} ms (±{latence_std:.2f})")
    
    # 7. Évaluation sur les données de test
    print("⏳ Évaluation sur SDNET2018...")
    
    # Prédictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Précision: {precision:.4f}")
    print(f"🎯 Rappel: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}  <--- MÉTRIQUE CLÉ")
    print(f"📊 Matrice de confusion:")
    print(f"   Vrais négatifs: {cm[0,0]}, Faux positifs: {cm[0,1]}")
    print(f"   Faux négatifs: {cm[1,0]}, Vrais positifs: {cm[1,1]}")
    
    # 8. Performance par catégorie (walls, decks, pavement)
    print("\n📊 Performance par type de structure:")
    for categorie in ['walls', 'decks', 'pavement']:
        indices = [i for i, s in enumerate(sources_test) if s == categorie]
        if indices:
            y_cat = y_test[indices]
            y_pred_cat = y_pred[indices]
            f1_cat = f1_score(y_cat, y_pred_cat, zero_division=0)
            print(f"   - {categorie}: F1 = {f1_cat:.4f} ({len(indices)} images)")
    
    # Sauvegarder les résultats
    resultats.append({
        "Modèle": modele_info["nom"],
        "Paramètres (M)": f"{params/1e6:.2f}",
        "Taille (Mo)": f"{taille_fichier:.2f}",
        "Latence (ms)": f"{latence_moyenne:.2f}",
        "Latence_std": f"{latence_std:.2f}",
        "Accuracy": f"{accuracy:.4f}",
        "Précision": f"{precision:.4f}",
        "Rappel": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}",
        "VN": cm[0,0],
        "FP": cm[0,1],
        "FN": cm[1,0],
        "VP": cm[1,1]
    })

# Afficher le tableau récapitulatif
print("\n" + "=" * 100)
print("📋 TABLEAU RÉCAPITULATIF DES RÉSULTATS SUR SDNET2018")
print("=" * 100)

print(f"\n{'Modèle':20} | {'Taille':8} | {'Latence':8} | {'F1-Score':8} | {'Précision':9} | {'Rappel':7}")
print("-" * 80)

for r in resultats:
    print(f"{r['Modèle']:20} | {r['Taille (Mo)']:>5} Mo | {r['Latence (ms)']:>6} ms | {r['F1-Score']:>8} | {r['Précision']:>9} | {r['Rappel']:>7}")

# Sauvegarder dans CSV
with open(fichier_resultats, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)

print(f"\n💾 Résultats détaillés sauvegardés dans : {fichier_resultats}")
print("\n✅ Profiling sur SDNET2018 terminé !")