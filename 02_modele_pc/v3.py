import tensorflow as tf
import numpy as np
import os
import cv2
import platform
import psutil
import cpuinfo
from glob import glob
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, matthews_corrcoef)
import pandas as pd
from tabulate import tabulate
import time
import subprocess
import json

print("=" * 90)
print("ÉVALUATION COMPLÈTE DE MOBILENETV3-SMALL SUR SDNET2018")
print("=" * 90)

# ============================================
# 1. INFORMATIONS ENVIRONNEMENT (PC)
# ============================================
print("\n" + "=" * 90)
print(" INFORMATIONS ENVIRONNEMENT DE TEST (PC)")
print("=" * 90)

# Informations CPU
cpu_info = cpuinfo.get_cpu_info()
print(f"\n  PROCESSEUR:")
print(f"   - Modèle: {cpu_info['brand_raw']}")
print(f"   - Architecture: {cpu_info['arch']}")
print(f"   - Cœurs physiques: {psutil.cpu_count(logical=False)}")
print(f"   - Cœurs logiques: {psutil.cpu_count(logical=True)}")
print(f"   - Fréquence: {psutil.cpu_freq().max:.0f} MHz")

# Informations RAM
mem = psutil.virtual_memory()
print(f"\n MÉMOIRE:")
print(f"   - RAM totale: {mem.total / (1024**3):.2f} GB")
print(f"   - RAM disponible: {mem.available / (1024**3):.2f} GB")

# Informations système
print(f"\n SYSTÈME:")
print(f"   - OS: {platform.system()} {platform.release()}")
print(f"   - Machine: {platform.node()}")
print(f"   - Python: {platform.python_version()}")
print(f"   - TensorFlow: {tf.__version__}")

# ============================================
# 2. INFORMATIONS TÉLÉPHONE (à remplir manuellement)
# ============================================
print("\n" + "=" * 90)
print(" INFORMATIONS TÉLÉPHONE (mesures réelles)")
print("=" * 90)

# À remplir avec les informations de TON téléphone
tel_info = {
    "Modele": "Redmi Note 11S  2201117SY",  # À MODIFIER
    "Processeur": "Helio G96 Octa-core Max 2.05Ghz",  # À MODIFIER
    "RAM": "6GB",  # À MODIFIER
    "Android": "13",  # À MODIFIER
    "Outil_mesure": "TensorFlow Lite Benchmark Tool + adb logcat"
}

print(f"\n INFORMATIONS TÉLÉPHONE:")
for key, value in tel_info.items():
    print(f"   - {key}: {value}")

# ============================================
# 3. MESURES TÉLÉPHONE DÉJÀ EFFECTUÉES
# ============================================
print(f"\n MESURES TÉLÉPHONE RÉELLES (benchmark effectué):")
print(f"   - Latence moyenne (4 threads): 11.2 ms")
print(f"   - Latence 1 thread: (à mesurer)")
print(f"   - Latence 2 threads: (à mesurer)")
print(f"   - Gain parallélisme (1→4): {(11.2 / 11.2):.2f}x (à recalculer avec mesures 1-2 threads)")
print(f"   - Profilage opérations: à intégrer depuis adb logcat")

# ============================================
# 4. CHARGEMENT DATASET
# ============================================
print("\n" + "=" * 90)
print(" ÉTAPE 1: Chargement des images de test (SDNE2018)")
print("=" * 90)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
CHEMIN_SDNET = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

def charger_images_test(chemin_base, max_images=500):
    """Charge les images depuis SDNET2018"""
    images = []
    labels = []
    sources = []
    
    categories = ['Walls', 'Decks', 'Pavements']
    
    print(f"\n Chargement depuis : {chemin_base}")
    
    for categorie in categories:
        # Images avec fissures
        chemin_cracked = os.path.join(chemin_base, categorie, 'cracked', '*.[jJ][pP][gG]')
        fichiers_cracked = glob(chemin_cracked)
        print(f"    {categorie}/cracked: {len(fichiers_cracked)} images")
        
        for img_path in fichiers_cracked[:max_images//6]:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)
                sources.append(categorie)
        
        # Images sans fissures
        chemin_nocracked = os.path.join(chemin_base, categorie, 'Non-cracked', '*.[jJ][pP][gG]')
        fichiers_nocracked = glob(chemin_nocracked)
        print(f"    {categorie}/Non-cracked: {len(fichiers_nocracked)} images")
        
        for img_path in fichiers_nocracked[:max_images//6]:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)
                sources.append(categorie)
    
    return np.array(images), np.array(labels), sources

X_test, y_test, sources = charger_images_test(CHEMIN_SDNET, max_images=500)
X_test = X_test / 255.0

print(f"\n Total chargé : {len(X_test)} images")
print(f"   - Avec fissures: {np.sum(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%)")
print(f"   - Sans fissures: {len(y_test)-np.sum(y_test)} ({(len(y_test)-np.sum(y_test))/len(y_test)*100:.1f}%)")

# ============================================
# 5. CHARGEMENT MODÈLE
# ============================================
print("\n" + "=" * 90)
print("🤖 ÉTAPE 2: Chargement de MobileNetV3-Small")
print("=" * 90)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.summary()

# Taille du modèle
model.save("temp_model.h5")
taille_modele = os.path.getsize("temp_model.h5") / (1024*1024)
os.remove("temp_model.h5")
print(f"\n Taille du modèle: {taille_modele:.2f} Mo")

# ============================================
# 6. MESURES PERFORMANCES PC
# ============================================
print("\n" + "=" * 90)
print(" ÉTAPE 3: Mesures de performances sur PC")
print("=" * 90)

# Latence
dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
for _ in range(10):
    model.predict(dummy_input, verbose=0)

latences = []
process = psutil.Process()
mem_avant = process.memory_info().rss / 1024 / 1024

for i in range(50):
    start = time.time()
    model.predict(dummy_input, verbose=0)
    end = time.time()
    latences.append((end - start) * 1000)

mem_apres = process.memory_info().rss / 1024 / 1024

latence_moyenne = np.mean(latences)
latence_std = np.std(latences)
latence_min = np.min(latences)
latence_max = np.max(latences)

print(f"\n LATENCE PC (50 itérations):")
print(f"   - Moyenne: {latence_moyenne:.2f} ms")
print(f"   - Écart-type: {latence_std:.2f} ms")
print(f"   - Minimum: {latence_min:.2f} ms")
print(f"   - Maximum: {latence_max:.2f} ms")
print(f"   - Mémoire utilisée: {mem_apres - mem_avant:.2f} MB")

# ============================================
# 7. ÉVALUATION CLASSIFICATION
# ============================================
print("\n" + "=" * 90)
print(" ÉTAPE 4: Évaluation classification sur SDNET2018")
print("=" * 90)

y_pred_proba = model.predict(X_test, verbose=1, batch_size=BATCH_SIZE)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
npv = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
balanced_acc = (recall + specificity) / 2
roc_auc = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)

# ============================================
# 8. AFFICHAGE RÉSULTATS
# ============================================
print("\n" + "=" * 90)
print(" RÉSULTATS DÉTAILLÉS - MOBILENETV3-SMALL")
print("=" * 90)

# Tableau 1: Métriques classification
print("\n🔹 MÉTRIQUES CLASSIFICATION (sur PC):")
tableau_metrics = [
    ["Métrique", "Valeur", "Interprétation"],
    ["Accuracy", f"{accuracy:.4f}", "Exactitude globale"],
    ["Précision", f"{precision:.4f}", "% de fissures correctement prédites"],
    ["Rappel (Sensitivity)", f"{recall:.4f}", "% de vraies fissures détectées"],
    ["F1-Score", f"{f1:.4f}", " MÉTRIQUE CLÉ (moyenne précision/rappel)"],
    ["Spécificité", f"{specificity:.4f}", "% d'images saines reconnues"],
    ["NPV", f"{npv:.4f}", "% de 'pas fissure' corrects"],
    ["Balanced Accuracy", f"{balanced_acc:.4f}", "Moyenne rappel/spécificité"],
    ["ROC-AUC", f"{roc_auc:.4f}", "Capacité de discrimination"],
    ["MCC", f"{mcc:.4f}", "Corrélation (-1 à 1)"],
]
print(tabulate(tableau_metrics[1:], headers=tableau_metrics[0], tablefmt="grid"))

# Tableau 2: Matrice de confusion
print("\n MATRICE DE CONFUSION:")
print(tabulate([["", "Prédit Négatif", "Prédit Positif"],
                ["Réel Négatif", cm[0,0], cm[0,1]],
                ["Réel Positif", cm[1,0], cm[1,1]]], 
               headers="firstrow", tablefmt="grid"))

# Tableau 3: Performances par structure
print("\n🔹 PERFORMANCES PAR TYPE DE STRUCTURE:")
struct_data = []
for cat in ['Walls', 'Decks', 'Pavements']:
    idx = [i for i, s in enumerate(sources) if s == cat]
    if idx:
        y_cat, y_pred_cat = y_test[idx], y_pred[idx]
        struct_data.append([
            cat, len(idx),
            f"{precision_score(y_cat, y_pred_cat, zero_division=0):.4f}",
            f"{recall_score(y_cat, y_pred_cat, zero_division=0):.4f}",
            f"{f1_score(y_cat, y_pred_cat, zero_division=0):.4f}"
        ])
print(tabulate(struct_data, 
               headers=["Structure", "Images", "Précision", "Rappel", "F1-Score"],
               tablefmt="grid"))

# ============================================
# 9. TABLEAU RÉCAPITULATIF COMPLET (PC + TÉLÉPHONE)
# ============================================
print("\n" + "=" * 90)
print(" TABLEAU RÉCAPITULATIF COMPLET - RÉPONSE AU PROF")
print("=" * 90)

# Création du tableau final
tableau_final = [
    ["Catégorie", "Métrique", "Valeur PC", "Valeur Téléphone", "Source"],
    
    ["ENVIRONNEMENT", "Processeur", cpu_info['brand_raw'], tel_info["Processeur"], "Spécifications"],
    ["", "Cœurs", f"{psutil.cpu_count(logical=True)} logiques", "4+", "Spécifications"],
    ["", "RAM", f"{mem.total / (1024**3):.2f} GB", tel_info["RAM"], "Spécifications"],
    ["", "OS", f"{platform.system()} {platform.release()}", f"Android {tel_info['Android']}", "Spécifications"],
    
    ["RESSOURCES", "Taille modèle", f"{taille_modele:.2f} Mo", "2.9 Mo (TFLite)", "Fichier"],
    ["", "Mémoire utilisée", f"{mem_apres - mem_avant:.2f} MB", "~10-15 MB (estimé)", "Mesure/Estimation"],
    
    ["PERFORMANCES", "Latence moyenne", f"{latence_moyenne:.2f} ms", "11.2 ms", "Script PC / adb logcat"],
    ["", "Latence min", f"{latence_min:.2f} ms", "-", "Script PC"],
    ["", "Latence max", f"{latence_max:.2f} ms", "-", "Script PC"],
    ["", "Écart-type", f"{latence_std:.2f} ms", "-", "Script PC"],
    
    ["CLASSIFICATION", "Accuracy", f"{accuracy:.4f}", "Non mesuré", "Script PC"],
    ["", "Précision", f"{precision:.4f}", "Non mesuré", "Script PC"],
    ["", "Rappel", f"{recall:.4f}", "Non mesuré", "Script PC"],
    ["", "F1-Score", f"{f1:.4f}", "Non mesuré", "Script PC "],
    ["", "Spécificité", f"{specificity:.4f}", "Non mesuré", "Script PC"],
    ["", "ROC-AUC", f"{roc_auc:.4f}", "Non mesuré", "Script PC"],
    ["", "MCC", f"{mcc:.4f}", "Non mesuré", "Script PC"],
    
    ["MATRICE", "VN", cm[0,0], "-", "Script PC"],
    ["", "FP", cm[0,1], "-", "Script PC"],
    ["", "FN", cm[1,0], "-", "Script PC"],
    ["", "VP", cm[1,1], "-", "Script PC"],
]

print(tabulate(tableau_final[1:], headers=tableau_final[0], tablefmt="grid", maxcolwidths=[15,20,15,15,20]))

# ============================================
# 10. INSTRUCTIONS POUR PROFILAGE TÉLÉPHONE
# ============================================
print("\n" + "=" * 90)
print(" INSTRUCTIONS POUR COMPLÉTER LES MESURES TÉLÉPHONE")
print("=" * 90)



# ============================================
# 11. SAUVEGARDE
# ============================================
print("\n" + "=" * 90)
print(" SAUVEGARDE DES RÉSULTATS")
print("=" * 90)

# Sauvegarde format CSV
df_final = pd.DataFrame(tableau_final[1:], columns=tableau_final[0])
df_final.to_csv('resultats_complets_mobilenetv3.csv', index=False, encoding='utf-8')
print(" Résultats sauvegardés dans 'resultats_complets_mobilenetv3.csv'")

# Sauvegarde format texte
with open('resultats_complets_mobilenetv3.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write("RÉSULTATS COMPLETS MOBILENETV3-SMALL\n")
    f.write("=" * 90 + "\n")
    f.write(tabulate(tableau_final[1:], headers=tableau_final[0], tablefmt="grid"))
print(" Résultats sauvegardés dans 'resultats_complets_mobilenetv3.txt'")

print("\n" + "=" * 90)
print(" ÉVALUATION TERMINÉE - TOUTES LES MÉTRIQUES PC SONT DISPONIBLES")
print("=" * 90)