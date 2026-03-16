import tensorflow as tf
import numpy as np
import os
import cv2
from glob import glob
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, matthews_corrcoef)
import pandas as pd
from tabulate import tabulate
import time
import psutil

print("=" * 80)
print("ÉVALUATION COMPLÈTE DE MOBILENETV3-SMALL SUR SDNET2018")
print("=" * 80)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
CHEMIN_SDNET = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

def charger_images_test(chemin_base, max_images=1000):
    """Charge les images depuis SDNET2018"""
    images = []
    labels = []
    sources = []  # Pour savoir d'où vient l'image (walls, decks, pavement)
    
    categories = ['Walls', 'Decks', 'Pavements']
    
    print(f"\n Chargement des images depuis : {chemin_base}")
    
    for categorie in categories:
        # Images avec fissures
        chemin_cracked = os.path.join(chemin_base, categorie, 'cracked', '*.[jJ][pP][gG]')
        fichiers_cracked = glob(chemin_cracked)
        print(f"   {categorie}/cracked: {len(fichiers_cracked)} images trouvées")
        
        for img_path in fichiers_cracked[:max_images//6]:
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
        chemin_nocracked = os.path.join(chemin_base, categorie, 'Non-cracked', '*.[jJ][pP][gG]')
        fichiers_nocracked = glob(chemin_nocracked)
        print(f"   {categorie}/Non-cracked: {len(fichiers_nocracked)} images trouvées")
        
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

# 1. CHARGEMENT DES DONNÉES
print("\n ÉTAPE 1: Chargement des images de test...")
X_test, y_test, sources = charger_images_test(CHEMIN_SDNET, max_images=500)

# Normalisation
X_test = X_test / 255.0

print(f"\n Total chargé : {len(X_test)} images")
print(f"   - Avec fissures: {np.sum(y_test)} images ({np.sum(y_test)/len(y_test)*100:.1f}%)")
print(f"   - Sans fissures: {len(y_test) - np.sum(y_test)} images ({(len(y_test)-np.sum(y_test))/len(y_test)*100:.1f}%)")
print(f"   - Répartition par type:")
print(f"     * Walls: {sources.count('Walls')} images")
print(f"     * Decks: {sources.count('Decks')} images")
print(f"     * Pavement: {sources.count('Pavements')} images")

# 2. CHARGEMENT DU MODÈLE
print("\n ÉTAPE 2: Chargement de MobileNetV3-Small...")

# Charger le modèle de base (sans la tête de classification)
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False

# Ajouter la tête de classification binaire
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Compiler
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Afficher le résumé du modèle
model.summary()

# 3. MESURE DE LA LATENCE SUR PC
print("\n  ÉTAPE 3: Mesure de la latence sur PC...")

# Warm-up
dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
for _ in range(10):
    model.predict(dummy_input, verbose=0)

# Mesure de la latence
latences = []
process = psutil.Process()
mem_avant = process.memory_info().rss / 1024 / 1024  # Mémoire avant en MB

for i in range(50):
    start = time.time()
    model.predict(dummy_input, verbose=0)
    end = time.time()
    latences.append((end - start) * 1000)  # en ms

mem_apres = process.memory_info().rss / 1024 / 1024

latence_moyenne = np.mean(latences)
latence_std = np.std(latences)
latence_min = np.min(latences)
latence_max = np.max(latences)

print(f"    Statistiques de latence (sur 50 itérations):")
print(f"      - Moyenne: {latence_moyenne:.2f} ms")
print(f"      - Écart-type: {latence_std:.2f} ms")
print(f"      - Minimum: {latence_min:.2f} ms")
print(f"      - Maximum: {latence_max:.2f} ms")
print(f"    Mémoire utilisée: {mem_apres - mem_avant:.2f} MB")

# 4. ÉVALUATION SUR SDNET2018
print("\n ÉTAPE 4: Évaluation sur SDNET2018...")

# Prédictions
print("    Prédictions en cours...")
y_pred_proba = model.predict(X_test, verbose=1, batch_size=BATCH_SIZE)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 5. CALCUL DE TOUTES LES MÉTRIQUES
print("\n ÉTAPE 5: Calcul des métriques...")

# Métriques de base
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Métriques avancées
specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
npv = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0  # Negative Predictive Value
ppv = precision  # Positive Predictive Value (same as precision)
fall_out = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0  # False Positive Rate
miss_rate = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0  # False Negative Rate
fpr = fall_out
tpr = recall

# Métriques supplémentaires
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
except:
    roc_auc = 0

try:
    mcc = matthews_corrcoef(y_test, y_pred)
except:
    mcc = 0

# Balanced accuracy
balanced_acc = (recall + specificity) / 2

# 6. AFFICHAGE DES RÉSULTATS
print("\n" + "=" * 80)
print(" RÉSULTATS DE L'ÉVALUATION - MOBILENETV3-SMALL")
print("=" * 80)

# Tableau 1: Métriques principales
print("\n MÉTRIQUES PRINCIPALES:")
tableau1 = [
    ["Métrique", "Valeur"],
    ["Accuracy (Exactitude)", f"{accuracy:.4f}"],
    ["Précision (PPV)", f"{precision:.4f}"],
    ["Rappel (Sensitivity/TPR)", f"{recall:.4f}"],
    ["F1-Score", f"{f1:.4f}   MÉTRIQUE CLÉ"],
    ["Spécificité (TNR)", f"{specificity:.4f}"],
    ["NPV", f"{npv:.4f}"],
]
print(tabulate(tableau1[1:], headers=tableau1[0], tablefmt="grid"))

# Tableau 2: Métriques avancées
print("\n MÉTRIQUES AVANCÉES:")
tableau2 = [
    ["Métrique", "Valeur"],
    ["Balanced Accuracy", f"{balanced_acc:.4f}"],
    ["ROC-AUC", f"{roc_auc:.4f}"],
    ["MCC (Matthews)", f"{mcc:.4f}"],
    ["False Positive Rate (FPR)", f"{fpr:.4f}"],
    ["False Negative Rate (FNR)", f"{miss_rate:.4f}"],
]
print(tabulate(tableau2[1:], headers=tableau2[0], tablefmt="grid"))

# Tableau 3: Matrice de confusion
print("\n MATRICE DE CONFUSION:")
print("                     Prédit")
print("                     Négatif   Positif")
print(f"Réel    Négatif     {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"        Positif     {cm[1,0]:6d}   {cm[1,1]:6d}")

# Tableau 4: Performances par catégorie
print("\n PERFORMANCES PAR TYPE DE STRUCTURE:")
print("Catégorie   | Images | F1-Score | Précision | Rappel")
print("-" * 55)

for categorie in ['Walls', 'Decks', 'Pavements']:
    indices = [i for i, s in enumerate(sources) if s == categorie]
    if indices:
        y_cat = y_test[indices]
        y_pred_cat = y_pred[indices]
        prec_cat = precision_score(y_cat, y_pred_cat, zero_division=0)
        rec_cat = recall_score(y_cat, y_pred_cat, zero_division=0)
        f1_cat = f1_score(y_cat, y_pred_cat, zero_division=0)
        print(f"{categorie:10} | {len(indices):5d} | {f1_cat:.4f}    | {prec_cat:.4f}   | {rec_cat:.4f}")

# Tableau 5: Rapport de classification complet
print("\ RAPPORT DE CLASSIFICATION COMPLET:")
print(classification_report(y_test, y_pred, target_names=['Pas de fissure', 'Fissure']))

# 7. COMPARAISON AVEC LA LATENCE TÉLÉPHONE
print("\n" + "=" * 80)
print(" COMPARAISON PC vs TÉLÉPHONE")
print("=" * 80)

print(f"""
RÉCAPITULATIF POUR MOBILENETV3-SMALL:

┌───────────────────────┬─────────────────┐
│ Métrique              │ Valeur          │
├───────────────────────┼─────────────────┤
│ Latence PC            │ {latence_moyenne:.2f} ms         │
│ Latence Téléphone     │ 11.2 ms         │
│ F1-Score              │ {f1:.4f}            │
│ Précision             │ {precision:.4f}            │
│ Rappel                │ {recall:.4f}            │
│ Accuracy              │ {accuracy:.4f}            │
│ Taille du modèle      │ 2.9 Mo          │
└───────────────────────┴─────────────────┘
""")

# 8. SAUVEGARDE DES RÉSULTATS
resultats = {
    "Modèle": "MobileNetV3-Small",
    "Latence_PC_ms": latence_moyenne,
    "Latence_Tel_ms": 11.2,
    "Accuracy": accuracy,
    "Précision": precision,
    "Rappel": recall,
    "F1-Score": f1,
    "Spécificité": specificity,
    "Balanced_Accuracy": balanced_acc,
    "ROC_AUC": roc_auc,
    "MCC": mcc,
    "Taille_Mo": 2.9,
    "VN": cm[0,0],
    "FP": cm[0,1],
    "FN": cm[1,0],
    "VP": cm[1,1]
}

# Sauvegarder en CSV
df_resultats = pd.DataFrame([resultats])
df_resultats.to_csv('resultats_mobilenetv3_complet.csv', index=False)
print("\n Résultats complets sauvegardés dans 'resultats_mobilenetv3_complet.csv'")

