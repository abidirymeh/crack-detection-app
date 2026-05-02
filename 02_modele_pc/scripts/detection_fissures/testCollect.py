# preparer_augmenter_tester.py
import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("=" * 70)
print("📊 PRÉPARATION, AUGMENTATION ET TEST SUR VOS IMAGES")
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

# 2. Fonction pour charger et redimensionner les images
def charger_et_redimensionner(dossier):
    images = []
    noms = []
    
    if not os.path.exists(dossier):
        print(f"   ⚠️ Dossier non trouvé: {dossier}")
        return images, noms
    
    for fichier in os.listdir(dossier):
        if fichier.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            chemin = os.path.join(dossier, fichier)
            img = cv2.imread(chemin)
            if img is not None:
                # Redimensionner à 224x224
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                noms.append(fichier)
                print(f"      ✅ Chargé: {fichier} ({img.shape[1]}x{img.shape[0]})")
            else:
                print(f"      ❌ Erreur: {fichier}")
    
    return images, noms

# 3. Fonction d'augmentation (8 versions)
def augmenter_image(img):
    """Génère 8 versions augmentées d'une image"""
    versions = []
    h, w = img.shape[:2]
    
    # 1. Originale
    versions.append(img)
    
    # 2. Rotation +10°
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1)
    rot10 = cv2.warpAffine(img, M, (w, h))
    versions.append(rot10)
    
    # 3. Rotation -10°
    M = cv2.getRotationMatrix2D((w/2, h/2), -10, 1)
    rot_10 = cv2.warpAffine(img, M, (w, h))
    versions.append(rot_10)
    
    # 4. Flip horizontal
    flip_h = cv2.flip(img, 1)
    versions.append(flip_h)
    
    # 5. Zoom (recadrage)
    zoom = cv2.resize(img[20:-20, 20:-20], (w, h))
    versions.append(zoom)
    
    # 6. Luminosité +30%
    bright = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
    versions.append(bright)
    
    # 7. Luminosité -30%
    dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)
    versions.append(dark)
    
    # 8. Léger flou
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    versions.append(blur)
    
    return versions

# 4. Charger vos images
print("\n📁 Chargement de vos images...")

print("\n📂 Dossier SAIN:")
X_sain, noms_sain = charger_et_redimensionner(
    r"C:\Users\RIMEH\Desktop\PFA_ADEVA\03_dataset\testImage\sain"
)

print("\n📂 Dossier FISSURE:")
X_fissure, noms_fissure = charger_et_redimensionner(
    r"C:\Users\RIMEH\Desktop\PFA_ADEVA\03_dataset\testImage\fissure"
)

print(f"\n📊 RÉSUMÉ:")
print(f"   Images saines trouvées: {len(X_sain)}")
print(f"   Images fissures trouvées: {len(X_fissure)}")

if len(X_sain) == 0 or len(X_fissure) == 0:
    print("\n❌ ERREUR: Aucune image trouvée !")
    print("   Vérifiez les chemins des dossiers.")
    exit()

# 5. Augmentation et test
print("\n🔄 Augmentation et test en cours...")

resultats = []

# Tester images saines
for i, img in enumerate(X_sain):
    versions = augmenter_image(img)
    probas = []
    
    for v in versions:
        v_norm = v / 255.0
        proba = predict_single(v_norm)
        probas.append(proba)
    
    resultats.append({
        'type': 'sain',
        'nom': noms_sain[i],
        'proba_originale': probas[0],
        'proba_moyenne': np.mean(probas),
        'proba_max': np.max(probas),
        'proba_min': np.min(probas),
        'nb_augmentations': len(versions)
    })
    print(f"   ✅ Sain traité: {noms_sain[i]} ({len(versions)} versions)")

# Tester images fissures
for i, img in enumerate(X_fissure):
    versions = augmenter_image(img)
    probas = []
    
    for v in versions:
        v_norm = v / 255.0
        proba = predict_single(v_norm)
        probas.append(proba)
    
    resultats.append({
        'type': 'fissure',
        'nom': noms_fissure[i],
        'proba_originale': probas[0],
        'proba_moyenne': np.mean(probas),
        'proba_max': np.max(probas),
        'proba_min': np.min(probas),
        'nb_augmentations': len(versions)
    })
    print(f"   ✅ Fissure traitée: {noms_fissure[i]} ({len(versions)} versions)")

# 6. Calculer les métriques
y_true = []
y_pred_orig = []
y_pred_moy = []
y_pred_max = []

for r in resultats:
    y_true.append(1 if r['type'] == 'fissure' else 0)
    y_pred_orig.append(1 if r['proba_originale'] > SEUIL else 0)
    y_pred_moy.append(1 if r['proba_moyenne'] > SEUIL else 0)
    y_pred_max.append(1 if r['proba_max'] > SEUIL else 0)

prec_orig = precision_score(y_true, y_pred_orig) if sum(y_pred_orig) > 0 else 0
rec_orig = recall_score(y_true, y_pred_orig)
f1_orig = f1_score(y_true, y_pred_orig)

prec_moy = precision_score(y_true, y_pred_moy) if sum(y_pred_moy) > 0 else 0
rec_moy = recall_score(y_true, y_pred_moy)
f1_moy = f1_score(y_true, y_pred_moy)

prec_max = precision_score(y_true, y_pred_max) if sum(y_pred_max) > 0 else 0
rec_max = recall_score(y_true, y_pred_max)
f1_max = f1_score(y_true, y_pred_max)

# 7. Afficher les résultats
print("\n" + "=" * 70)
print("📊 RÉSULTATS COMPARATIFS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    MÉTRIQUES SUR VOS IMAGES                              │
├───────────────────────┬─────────────┬───────────┬───────────────────────┤
│       Version         │  Précision  │   Recall  │       F1-Score        │
├───────────────────────┼─────────────┼───────────┼───────────────────────┤
│ Images originales     │   {prec_orig:.2%}    │   {rec_orig:.2%}   │        {f1_orig:.2%}         │
│ Moyenne (8 versions)  │   {prec_moy:.2%}    │   {rec_moy:.2%}   │        {f1_moy:.2%}         │
│ Maximum (8 versions)  │   {prec_max:.2%}    │   {rec_max:.2%}   │        {f1_max:.2%}         │
└───────────────────────┴─────────────┴───────────┴───────────────────────┘
""")

# 8. Matrice de confusion
print("\n" + "=" * 70)
print("📊 MATRICE DE CONFUSION (images originales)")
print("=" * 70)

y_pred_orig_bin = np.array(y_pred_orig)
y_true_bin = np.array(y_true)
cm = confusion_matrix(y_true_bin, y_pred_orig_bin)

print(f"""
                 Prédiction
              Sain    Fissure
Réel Sain      {cm[0][0]:>3}        {cm[0][1]:>3}
Réel Fissure   {cm[1][0]:>3}        {cm[1][1]:>3}
""")

# 9. Détail par image
print("\n" + "=" * 70)
print("📋 DÉTAIL PAR IMAGE (probabilités originales)")
print("=" * 70)

print("\n🔹 IMAGES SAINES (doivent être < 0.57):")
for r in resultats:
    if r['type'] == 'sain':
        statut = "✅ CORRECT" if r['proba_originale'] < SEUIL else "❌ FAUX POSITIF"
        print(f"   {r['nom']:<40} proba={r['proba_originale']:.3f} → {statut}")

print("\n🔹 IMAGES FISSURES (doivent être > 0.57):")
for r in resultats:
    if r['type'] == 'fissure':
        statut = "✅ CORRECT" if r['proba_originale'] > SEUIL else "❌ NON DÉTECTÉE"
        print(f"   {r['nom']:<40} proba={r['proba_originale']:.3f} → {statut}")

# 10. Sauvegarder les images augmentées
print("\n" + "=" * 70)
print("💾 SAUVEGARDE DES IMAGES AUGMENTÉES")
print("=" * 70)

reponse = input("\nVoulez-vous sauvegarder les images augmentées ? (o/n): ")
if reponse.lower() == 'o':
    # Créer les dossiers
    os.makedirs('augmented/sain', exist_ok=True)
    os.makedirs('augmented/fissure', exist_ok=True)
    
    # Sauvegarder images saines augmentées
    for i, img in enumerate(X_sain):
        versions = augmenter_image(img)
        nom_base = os.path.splitext(noms_sain[i])[0]
        for j, v in enumerate(versions):
            cv2.imwrite(f'augmented/sain/{nom_base}_aug{j}.jpg', cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    
    # Sauvegarder images fissures augmentées
    for i, img in enumerate(X_fissure):
        versions = augmenter_image(img)
        nom_base = os.path.splitext(noms_fissure[i])[0]
        for j, v in enumerate(versions):
            cv2.imwrite(f'augmented/fissure/{nom_base}_aug{j}.jpg', cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    
    print("✅ Images augmentées sauvegardées dans le dossier 'augmented/'")

# 11. Résumé pour le jury
print("\n" + "=" * 70)
print("🎯 RÉSUMÉ POUR LE JURY")
print("=" * 70)

print(f"""
Sur un dataset personnel de {len(X_sain)} images saines et {len(X_fissure)} images fissurées,
après augmentation (8 versions par image), voici les performances de mon modèle :

┌─────────────────────────────────────────────────────────────────────────┐
│  • Précision originale : {prec_orig:.1%}                                          │
│  • Recall original    : {rec_orig:.1%}                                          │
│  • F1-Score original  : {f1_orig:.1%}                                          │
│                                                                          │
│  • Après augmentation (moyenne) :                                      │
│    Précision : {prec_moy:.1%}, Recall : {rec_moy:.1%}, F1 : {f1_moy:.1%}                │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ Test terminé !")
