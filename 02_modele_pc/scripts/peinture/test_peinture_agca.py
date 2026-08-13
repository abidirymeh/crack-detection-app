# test_peinture_agca.py
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

print("=" * 60)
print("🎨 TEST MODÈLE PEINTURE AVEC AGCA")
print("=" * 60)

# Charger le nouveau modèle
interpreter = tf.lite.Interpreter('modele_peinture_agca.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# Charger le dataset de test
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

# Prédictions
print("\n🔄 Calcul des prédictions...")
probas = []
for i, chemin in enumerate(X_test):
    probas.append(predict(chemin))
    if (i+1) % 100 == 0:
        print(f"   → {i+1}/{len(X_test)} images")

probas = np.array(probas)
y_true = np.array(y_test)

# Résultats
print("\n" + "=" * 60)
print("📊 RÉSULTATS DU MODÈLE AVEC AGCA")
print("=" * 60)

# Meilleur seuil
best_f1 = 0
best_seuil = 0.5
best_prec = 0
best_rec = 0

for seuil in np.arange(0.40, 0.85, 0.01):
    y_pred = (probas > seuil).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_seuil = seuil
        best_prec = precision_score(y_true, y_pred)
        best_rec = recall_score(y_true, y_pred)

print(f"\n🔧 Seuil optimal: {best_seuil:.2f}")
print(f"   🎯 Précision: {best_prec:.2%}")
print(f"   🔍 Recall: {best_rec:.2%}")
print(f"   ⭐ F1: {best_f1:.2%}")

# Matrice de confusion
y_pred_final = (probas > best_seuil).astype(int)
cm = confusion_matrix(y_true, y_pred_final)

print(f"\n📊 Matrice de confusion:")
print(f"               Prédit OK    Prédit ABÎMÉE")
print(f"Réel OK          {cm[1][1]:>3}          {cm[1][0]:>3}")
print(f"Réel ABÎMÉE      {cm[0][1]:>3}          {cm[0][0]:>3}")

print("\n✅ Test terminé !")