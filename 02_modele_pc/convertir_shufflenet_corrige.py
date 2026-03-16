import tensorflow as tf
import kagglehub

print("🔄 Conversion de ShuffleNetV2 via KaggleHub...")

# Télécharger le modèle ShuffleNetV2 depuis Kaggle
model_path = kagglehub.model_download('tensorflow/shufflenet-v2/tfLite/default')

# Charger le modèle TFLite directement (plus simple)
interpreter = tf.lite.Interpreter(model_path=model_path + '/1.tflite')
interpreter.allocate_tensors()

# Sauvegarder une copie locale
with open("ShuffleNetV2.tflite", "wb") as f:
    f.write(open(model_path + '/1.tflite', 'rb').read())

print("✅ ShuffleNetV2.tflite créé !")
print(f"📁 Modèle téléchargé depuis : {model_path}")