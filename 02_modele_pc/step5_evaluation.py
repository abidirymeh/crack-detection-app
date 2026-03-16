# step5_evaluation_tflite_pretrained.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 70)
print("ÉVALUATION AVEC MODÈLE TFLITE PRÉ-ENTRAÎNÉ")
print("=" * 70)

# Charger les données
data = np.load('sdnet_prepared.npz')
X_test, y_test = data['X_test'], data['y_test']
print(f"✅ {len(X_test)} images de test")

# Charger TFLite
interpreter = tf.lite.Interpreter(model_path='MobileNetV3-Small.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prédictions
y_pred = []
for i in range(len(X_test)):
    input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(1 if output[0][0] > 0.5 else 0)

# Résultats
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📌 Matrice de confusion:")
print(f"{'':15} {'Prédit SAIN':15} {'Prédit FISSURE':15}")
print(f"{'Vrai SAIN':15} {tn:15} {fp:15}")
print(f"{'Vrai FISSURE':15} {fn:15} {tp:15}")

accuracy = (tn + tp) / (tn + fp + fn + tp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📈 Métriques:")
print(f"   - Accuracy: {accuracy:.4f}")
print(f"   - Précision: {precision:.4f}")
print(f"   - Rappel: {recall:.4f}")
print(f"   - F1: {f1:.4f}")