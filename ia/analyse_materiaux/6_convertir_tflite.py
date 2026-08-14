# convertir_materiaux.py
import tensorflow as tf
import numpy as np

print("=" * 60)
print(" CONVERSION MODÈLE MATÉRIAUX EN TFLite")
print("=" * 60)

# 1 Charger le modèle entraîné sur UCI Concrete
print("\n Chargement du modèle...")
model = tf.keras.models.load_model('modele_concrete_final.h5')
print(" Modèle chargé")

# 2 Vérifier la forme d'entrée
print(f"\n Forme d'entrée attendue: {model.input_shape}")
print(f" Forme de sortie: {model.output_shape}")

# 3 Convertir en TFLite
print("\n Conversion en TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 4 Sauvegarder
with open('modele_concrete.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n Modèle TFLite créé: modele_concrete.tflite")
print(f"   Taille: {len(tflite_model) / (1024*1024):.2f} Mo")

# 5 Test rapide
print("\n Test rapide...")
interpreter = tf.lite.Interpreter(model_path='modele_concrete.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Entrée: {input_details[0]['shape']}")
print(f"   Sortie: {output_details[0]['shape']}")
print(" Modèle valide")

print("\n Copie dans Android:")
print("   copy modele_concrete.tflite C:\\Users\\RIMEH\\AndroidStudioProjects\\DetectionFissuresApp22\\app\\src\\main\\assets\\")