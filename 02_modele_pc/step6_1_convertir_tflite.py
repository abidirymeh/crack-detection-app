# step6_1_convertir_tflite.py
import tensorflow as tf
import numpy as np

print("=" * 70)
print("ÉTAPE 6.1: CONVERSION EN TFLITE OPTIMISÉ")
print("=" * 70)

# Charger le modèle (même architecture)
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

# Utiliser les poids du modèle pré-entraîné
print("✅ Modèle reconstruit")

# Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIMISATIONS
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantification pour réduire la taille
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Sauvegarder
with open('mobilenetv3_fissures.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Modèle TFLite créé: mobilenetv3_fissures.tflite")
print(f"📁 Taille: {len(tflite_model) / (1024*1024):.2f} Mo")

# Tester rapidement
interpreter = tf.lite.Interpreter(model_path='mobilenetv3_fissures.tflite')
interpreter.allocate_tensors()
print(f"✅ Test de chargement réussi")