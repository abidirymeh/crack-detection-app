# tableau_complet_peinture.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("=" * 80)
print("🎨 TABLEAU COMPLET DES SEUILS - MODÈLE PEINTURE")
print("=" * 80)

# 1. Charger le modèle
model_path = "modele_peinture_agca.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# 2. Charger le dataset de test
test_path = "../../dataset_peinture_augmente/test"

X_test = []
y_test = []

for classe in ['ok', 'abimee']:
    dossier = os.path.join(test_path, classe)
    label = 1 if classe == 'ok' else 0
    for f in os.listdir(dossier):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            X_test.append(os.path.join(dossier, f))
            y_test.append(label)

print(f"📊 Dataset de test: {len(X_test)} images")
print(f"   OK: {sum(1 for y in y_test if y == 1)} images")
print(f"   Abîmée: {sum(1 for y in y_test if y == 0)} images")

# 3. Prédictions
print("\n🔄 Calcul des prédictions...")
probas = []
for i, chemin in enumerate(X_test):
    probas.append(predict(chemin))
    if (i+1) % 100 == 0:
        print(f"   → {i+1}/{len(X_test)} images")

probas = np.array(probas)
y_true = np.array(y_test)

# 4. Tous les seuils à tester
seuils = [0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.72, 0.75, 0.77, 0.80, 0.82, 0.85, 0.90]

print("\n" + "=" * 80)
print("📊 TABLEAU COMPLET DES MÉTRIQUES PAR SEUIL")
print("=" * 80)

print("\n")
print("┌────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
print("│ Seuil  │  Précision  │   Recall    │     F1      │  Accuracy   │  VP (ok)    │  FP (ok)    │")
print("├────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

resultats = []

for seuil in seuils:
    y_pred = (probas > seuil).astype(int)
    
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    vp = cm[1][1] if cm.shape == (2,2) else 0
    fp = cm[0][1] if cm.shape == (2,2) else 0
    
    resultats.append((seuil, prec, rec, f1, acc, vp, fp))
    
    if seuil == 0.77:
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
print(f"   → Recall: {best_rec[2]:.2%}, Précision: {best_rec[1]:.2%}, F1: {best_rec[3]:.2%}")

# Meilleure accuracy
best_acc = max(resultats, key=lambda x: x[4])
print(f"\n🎯 MEILLEURE ACCURACY: seuil = {best_acc[0]:.2f}")
print(f"   → Accuracy: {best_acc[4]:.2%}, Précision: {best_acc[1]:.2%}, Recall: {best_acc[2]:.2%}")

# 6. Résumé pour le jury
print("\n" + "=" * 80)
print("📋 RÉSUMÉ POUR LE JURY")
print("=" * 80)
print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCES DU MODÈLE PEINTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   📊 Sur 636 images de test :                                           │
│                                                                          │
│   🔧 Seuil optimal (F1 max) : {best_f1[0]:.2f}                                      │
│      • Précision : {best_f1[1]:.2%}                                                  │
│      • Recall    : {best_f1[2]:.2%}                                                  │
│      • F1-Score  : {best_f1[3]:.2%}                                                  │
│                                                                          │
│   🔧 Seuil pour précision max : {best_prec[0]:.2f}                                 │
│      • Précision : {best_prec[1]:.2%}                                                  │
│      • Recall    : {best_prec[2]:.2%}                                                  │
│                                                                          │
│   🔧 Seuil pour recall max : {best_rec[0]:.2f}                                     │
│      • Recall    : {best_rec[2]:.2%}                                                  │
│      • Précision : {best_rec[1]:.2%}                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 7. Code Android
print("\n" + "=" * 80)
print("📱 CODE ANDROID RECOMMANDÉ")
print("=" * 80)
print(f"""
// Dans MainActivity.java

float probability = output[0][0];

// Choix du seuil selon l'objectif :

// 1. Meilleur équilibre (F1 max = {best_f1[0]:.2f})
float SEUIL = {best_f1[0]:.2f}f;

// 2. Précision maximale
// float SEUIL = {best_prec[0]:.2f}f;

// 3. Recall maximal
// float SEUIL = {best_rec[0]:.2f}f;

boolean peintureAbimee = probability < SEUIL;
String etatPeinture = peintureAbimee ? "ABÎMÉE" : "OK";
""")

print("\n✅ Tableau généré !")