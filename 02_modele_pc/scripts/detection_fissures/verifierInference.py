import time
import numpy as np
import tensorflow as tf

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter('modele_enriched.tflite')
interpreter.allocate_tensors()

# Mesurer le temps
start = time.time()
for _ in range(100):
    interpreter.invoke()
end = time.time()
print(f"Temps moyen: {(end-start)/100*1000:.2f} ms")