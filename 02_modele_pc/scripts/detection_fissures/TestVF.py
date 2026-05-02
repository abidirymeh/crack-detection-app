# test_augmented_complete.py
import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("=" * 80)
print("📊 TEST COMPLET SUR LES IMAGES AUGMENTÉES - TOUS LES SEUILS")
print("=" * 80)

# Configuration
IMG_SIZE = 224

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

# 2. Charger les images augmentées
def charger_images_augmentees(dossier):
    images = []
    if not os.path.exists(dossier):
        return images
    for fichier in os.listdir(dossier):
        if fichier.lower().endswith(('.jpg', '.png', '.jpeg')):
            chemin = os.path.join(dossier, fichier)
            img = cv2.imread(chemin)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images.append(img)
    return np.array(images)

print("\n📁 Chargement des images augmentées...")
X_sain = charger_images_augmentees('augmented/sain')
X_fissure = charger_images_augmentees('augmented/fissure')
print(f"   Images saines: {len(X_sain)}")
print(f"   Images fissures: {len(X_fissure)}")
print(f"   TOTAL: {len(X_sain) + len(X_fissure)} images")

# 3. Prédictions
print("\n🔄 Prédictions en cours...")
probas_sain = []
probas_fissure = []

print("   Test des images saines...")
for i, img in enumerate(X_sain):
    probas_sain.append(predict_single(img))
    if (i+1) % 20 == 0:
        print(f"      → {i+1}/{len(X_sain)}")

print("   Test des images fissures...")
for i, img in enumerate(X_fissure):
    probas_fissure.append(predict_single(img))
    if (i+1) % 50 == 0:
        print(f"      → {i+1}/{len(X_fissure)}")

probas_sain = np.array(probas_sain)
probas_fissure = np.array(probas_fissure)

# 4. Tous les seuils à tester
seuils = [0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.70]

print("\n" + "=" * 80)
print("📊 RÉSULTATS POUR TOUS LES SEUILS")
print("=" * 80)

print("\n")
print("┌────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
print("│ Seuil  │  Précision  │   Recall    │     F1      │  Accuracy   │  VP (fiss)  │  FP (sain)  │")
print("├────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

resultats = []

for seuil in seuils:
    # Prédictions pour chaque seuil
    y_pred_sain = (probas_sain > seuil).astype(int)
    y_pred_fissure = (probas_fissure > seuil).astype(int)
    
    # Fusionner
    y_true = [0] * len(probas_sain) + [1] * len(probas_fissure)
    y_pred = list(y_pred_sain) + list(y_pred_fissure)
    
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    vp = cm[1][1]
    fp = cm[0][1]
    
    resultats.append((seuil, prec, rec, f1, acc, vp, fp))
    
    # Mettre en évidence le seuil actuel (0.57)
    if seuil == 0.57:
        print(f"│ \033[92m{seuil:.2f}  \033[0m │ \033[92m {prec:.2%}   \033[0m │ \033[92m {rec:.2%}   \033[0m │ \033[92m {f1:.2%}   \033[0m │ \033[92m {acc:.2%}   \033[0m │ \033[92m   {vp:>3}    \033[0m │ \033[92m   {fp:>3}    \033[0m │")
    else:
        print(f"│ {seuil:.2f}  │   {prec:.2%}   │   {rec:.2%}   │   {f1:.2%}   │   {acc:.2%}   │    {vp:>3}    │    {fp:>3}    │")

print("└────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

# 5. Meilleurs seuils par objectif
print("\n" + "=" * 80)
print("🏆 MEILLEURS SEUILS PAR OBJECTIF")
print("=" * 80)

# Meilleur F1
best_f1 = max(resultats, key=lambda x: x[3])
print(f"\n🎯 MEILLEUR F1: seuil = {best_f1[0]:.2f}")
print(f"   → Précision: {best_f1[1]:.2%}, Recall: {best_f1[2]:.2%}, F1: {best_f1[3]:.2%}")

# Meilleure précision
best_prec = max(resultats, key=lambda x: x[1])
print(f"\n🎯 MEILLEURE PRÉCISION: seuil = {best_prec[0]:.2f}")
print(f"   → Précision: {best_prec[1]:.2%}, Recall: {best_prec[2]:.2%}, F1: {best_prec[3]:.2%}")

# Meilleur recall
best_rec = max(resultats, key=lambda x: x[2])
print(f"\n🎯 MEILLEUR RECALL: seuil = {best_rec[0]:.2f}")
print(f"   → Précision: {best_rec[1]:.2%}, Recall: {best_rec[2]:.2%}, F1: {best_rec[3]:.2%}")

# 6. Seuil recommandé
print("\n" + "=" * 80)
print("📱 RECOMMANDATION POUR ANDROID")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEUIL RECOMMANDÉ                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Pour votre application Android, voici les choix possibles :            │
│                                                                          │
│  1. ÉQUILIBRE MAXIMUM (F1 max) :                                        │
│     → Seuil = {best_f1[0]:.2f}                                                  │
│     → Précision: {best_f1[1]:.2%}, Recall: {best_f1[2]:.2%}, F1: {best_f1[3]:.2%}        │
│                                                                          │
│  2. PRÉCISION MAXIMALE (peu de fausses alertes) :                       │
│     → Seuil = {best_prec[0]:.2f}                                                 │
│     → Précision: {best_prec[1]:.2%}, Recall: {best_prec[2]:.2%}                   │
│                                                                          │
│  3. RECALL MAXIMUM (détecter un max de fissures) :                      │
│     → Seuil = {best_rec[0]:.2f}                                                 │
│     → Recall: {best_rec[2]:.2%}, Précision: {best_rec[1]:.2%}                    │
│                                                                          │
│  4. VOTRE SEUIL ACTUEL (0.57) :                                         │
│     → Précision: 78.92%, Recall: 48.67%                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 7. Code Android recommandé
print("\n" + "=" * 80)
print("📱 CODE ANDROID RECOMMANDÉ")
print("=" * 80)
print(f"""
// Dans MainActivity.java - Version optimisée

float probability = output[0][0];

// Selon votre objectif, choisissez le seuil :

// Option 1: Meilleur équilibre (F1 max = {best_f1[0]:.2f})
float SEUIL = {best_f1[0]:.2f}f;

// Option 2: Précision maximale (peu de fausses alertes)
// float SEUIL = {best_prec[0]:.2f}f;

// Option 3: Recall maximal (détecter un max de fissures)
// float SEUIL = {best_rec[0]:.2f}f;

// Option 4: Votre seuil actuel
// float SEUIL = 0.57f;

boolean isCracked = probability > SEUIL;

// Interface utilisateur
if (probability > 0.65) {{
    resultText.setText("🔴 FISSURE CONFIRMÉE");
}} else if (probability > SEUIL) {{
    resultText.setText("🟡 FISSURE PROBABLE");
}} else {{
    resultText.setText("🟢 SURFACE SAINE");
}}
""")

# 8. Générer un graphique (optionnel)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    seuils_plot = [r[0] for r in resultats]
    prec_plot = [r[1] * 100 for r in resultats]
    rec_plot = [r[2] * 100 for r in resultats]
    f1_plot = [r[3] * 100 for r in resultats]
    
    plt.plot(seuils_plot, prec_plot, 'b-o', label='Précision', linewidth=2, markersize=8)
    plt.plot(seuils_plot, rec_plot, 'r-o', label='Recall', linewidth=2, markersize=8)
    plt.plot(seuils_plot, f1_plot, 'g-o', label='F1-Score', linewidth=2, markersize=8)
    
    plt.axvline(x=0.57, color='black', linestyle='--', label='Votre seuil (0.57)', alpha=0.7)
    plt.axvline(x=best_f1[0], color='gold', linestyle='--', label=f'Meilleur F1 ({best_f1[0]:.2f})', alpha=0.7)
    
    plt.xlabel('Seuil de décision', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Performance du modèle selon le seuil - Images augmentées', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('performance_seuils.png', dpi=150)
    print("\n✅ Graphique sauvegardé: performance_seuils.png")
except:
    print("\n⚠️ Graphique non généré (matplotlib non installé)")

print("\n✅ Test terminé !")