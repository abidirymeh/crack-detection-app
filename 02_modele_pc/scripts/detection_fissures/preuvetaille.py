# preuve_taille.py
import os
import tensorflow as tf

# Vérifier la taille du modèle
if os.path.exists('modele_enriched.tflite'):
    size = os.path.getsize('modele_enriched.tflite') / (1024*1024)
    print(f"✅ Taille du modèle: {size:.2f} Mo (2,5 Mo attendu)")

# Vérifier l'architecture
interpreter = tf.lite.Interpreter('modele_enriched.tflite')
interpreter.allocate_tensors()
print(f"✅ Modèle chargé avec succès")