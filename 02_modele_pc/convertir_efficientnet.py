import tensorflow as tf

print("🔄 Conversion de EfficientNet-B0...")

# Charger EfficientNet-B0
model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# Ajouter la couche de classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = model(inputs, training=False)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model_complet = tf.keras.Model(inputs, outputs)

# Convertir en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_complet)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Sauvegarder
with open("EfficientNet-B0.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ EfficientNet-B0.tflite créé !")
print(f"📁 Taille: {len(tflite_model) / (1024*1024):.2f} Mo")