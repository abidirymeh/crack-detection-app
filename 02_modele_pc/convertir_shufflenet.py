import tensorflow as tf
import tensorflow_hub as hub

print("🔄 Conversion de ShuffleNetV2 depuis TensorFlow Hub...")

# Charger ShuffleNetV2 depuis TF Hub
model_url = "https://tfhub.dev/tensorflow/tfjs-model/shufflenet_v2/1/default/1"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Sauvegarder
with open("ShuffleNetV2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ ShuffleNetV2.tflite créé !")
print(f"📁 Taille: {len(tflite_model) / (1024*1024):.2f} Mo")