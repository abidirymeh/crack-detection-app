# tester_augmented.py
import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("=" * 70)
print("📊 TEST SUR LES 280 IMAGES AUGMENTÉES")
print("=" * 70)

# Configuration
IMG_SIZE = 224
SEUIL = 0.57  # Votre seuil Android

# 1. Charger votre modèle
print("\n🤖 Chargement du modèle...")
interpreter = tf.lite.Interpreter('modele_enriched.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_single(img):
    inp = np.expand_dims(img, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# 2. Fonction pour charger les images augmentées
def charger_images_augmentees(dossier):
    images = []
    noms = []
    
    if not os.path.exists(dossier):
        print(f"   ⚠️ Dossier non trouvé: {dossier}")
        return images, noms
    
    for fichier in os.listdir(dossier):
        if fichier.lower().endswith(('.jpg', '.png', '.jpeg')):
            chemin = os.path.join(dossier, fichier)
            img = cv2.imread(chemin)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images.append(img)
                noms.append(fichier)
    
    return np.array(images), noms

# 3. Charger les images augmentées
print("\n📁 Chargement des images augmentées...")

X_sain, noms_sain = charger_images_augmentees('augmented/sain')
X_fissure, noms_fissure = charger_images_augmentees('augmented/fissure')

print(f"   Images saines augmentées: {len(X_sain)}")
print(f"   Images fissures augmentées: {len(X_fissure)}")
print(f"   TOTAL: {len(X_sain) + len(X_fissure)} images")

# 4. Prédictions
print("\n🔄 Prédictions en cours...")

y_true = []
y_pred = []
probas = []

# Tester images saines augmentées (label = 0)
print("   Test des images saines...")
for i, img in enumerate(X_sain):
    proba = predict_single(img)
    y_true.append(0)
    y_pred.append(1 if proba > SEUIL else 0)
    probas.append(proba)
    if (i+1) % 50 == 0:
        print(f"      → {i+1}/{len(X_sain)}")

# Tester images fissures augmentées (label = 1)
print("   Test des images fissures...")
for i, img in enumerate(X_fissure):
    proba = predict_single(img)
    y_true.append(1)
    y_pred.append(1 if proba > SEUIL else 0)
    probas.append(proba)
    if (i+1) % 50 == 0:
        print(f"      → {i+1}/{len(X_fissure)}")

# 5. Métriques
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 70)
print("📊 RÉSULTATS SUR LES 280 IMAGES AUGMENTÉES")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    MÉTRIQUES SUR 280 IMAGES                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   🎯 PRÉCISION : {prec:.2%} ({prec*100:.1f}%)                                         │
│   🔍 RECALL    : {rec:.2%} ({rec*100:.1f}%)                                         │
│   ⭐ F1-SCORE  : {f1:.2%} ({f1*100:.1f}%)                                         │
│   📈 ACCURACY  : {acc:.2%} ({acc*100:.1f}%)                                         │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                      MATRICE DE CONFUSION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                     Prédit Sain    Prédit Fissure                        │
│   Réel Sain           {cm[0][0]:>4}              {cm[0][1]:>4}                                │
│   Réel Fissure         {cm[1][0]:>4}              {cm[1][1]:>4}                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 6. Statistiques détaillées
print("\n" + "=" * 70)
print("📊 STATISTIQUES DÉTAILLÉES")
print("=" * 70)

print(f"\n🔹 Images SAINES augmentées ({len(X_sain)}):")
nb_correct_sain = cm[0][0]
nb_faux_positifs = cm[0][1]
print(f"   ✅ Correctement classées: {nb_correct_sain} ({nb_correct_sain/len(X_sain)*100:.1f}%)")
print(f"   ❌ Faux positifs (classées fissure): {nb_faux_positifs} ({nb_faux_positifs/len(X_sain)*100:.1f}%)")

print(f"\n🔹 Images FISSURES augmentées ({len(X_fissure)}):")
nb_correct_fissure = cm[1][1]
nb_faux_negatifs = cm[1][0]
print(f"   ✅ Correctement classées: {nb_correct_fissure} ({nb_correct_fissure/len(X_fissure)*100:.1f}%)")
print(f"   ❌ Non détectées (classées sain): {nb_faux_negatifs} ({nb_faux_negatifs/len(X_fissure)*100:.1f}%)")

# 7. Comparaison avec les résultats précédents
print("\n" + "=" * 70)
print("📈 COMPARAISON AVEC LES IMAGES ORIGINALES")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPARAISON                                           │
├─────────────────────────────┬─────────────┬───────────┬─────────────────┤
│          Version            │  Précision  │   Recall  │     F1          │
├─────────────────────────────┼─────────────┼───────────┼─────────────────┤
│ Images originales (35)      │   92.86%    │  86.67%   │    89.66%       │
│ Images augmentées (280)     │   {prec:.2%}    │   {rec:.2%}   │    {f1:.2%}        │
└─────────────────────────────┴─────────────┴───────────┴─────────────────┘
""")

# 8. Interprétation
print("\n" + "=" * 70)
print("🎯 INTERPRÉTATION")
print("=" * 70)

if prec > 0.85 and rec > 0.85:
    print("\n✅ EXCELLENT ! Votre modèle fonctionne parfaitement sur les images augmentées.")
elif prec > 0.75 and rec > 0.75:
    print("\n👍 TRÈS BON ! Votre modèle fonctionne bien sur les images augmentées.")
elif prec > 0.65:
    print("\n📊 CORRECT. Le modèle se comporte bien sur les variations.")
else:
    print("\n⚠️ Les augmentations ont dégradé les performances. À analyser.")

# 9. Code Android
print("\n" + "=" * 70)
print("📱 CODE ANDROID RECOMMANDÉ")
print("=" * 70)
print(f"""
// Dans MainActivity.java
float probability = output[0][0];
float SEUIL = {SEUIL}f;
boolean isCracked = probability > SEUIL;
""")

print("\n✅ Test terminé !")