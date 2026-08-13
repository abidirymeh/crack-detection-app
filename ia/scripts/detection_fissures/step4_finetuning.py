# step4_finetuning_mobilenetv2.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("=" * 60)
print("🚀 ENTRAÎNEMENT AVEC MOBILENETV2 (STABLE)")
print("=" * 60)

# 1. Charger les données
print("\n📥 Chargement des données...")
data = np.load('sdnet_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print(f"Train: {len(X_train)} images")
print(f"Test: {len(X_test)} images")
print(f"Sain (0): {np.sum(y_train==0)} train, {np.sum(y_test==0)} test")
print(f"Fissure (1): {np.sum(y_train==1)} train, {np.sum(y_test==1)} test")

# 2. Data augmentation
print("\n🔄 Data augmentation...")
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
datagen.fit(X_train)

# 3. MODÈLE MOBILENETV2 (STABLE !)
print("\n🏗️ Création du modèle MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# Geler les premières couches
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Architecture simple et efficace
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
)

print(f"✅ Modèle créé: {model.count_params():,} paramètres")

# 4. Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_recall',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'modele_mobilenetv2.h5',
        monitor='val_recall',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# 5. Entraînement
print("\n" + "=" * 60)
print("🔥 ENTRAÎNEMENT - 50 ÉPOQUES")
print("=" * 60)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# 6. Évaluation
print("\n📊 ÉVALUATION:")
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Sain', 'Fissure']))

# 7. Sauvegarde
model.save('mobilenetv2_fissures_final.h5')

# 8. Conversion TFLite (SANS FLEX OPS !)
# 8. Conversion TFLite (avec Flex ops)
print("\n🔄 Conversion en TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔴 LES 3 LIGNES À AJOUTER (ni plus, ni moins)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = []

tflite_model = converter.convert()

with open('mobilenetv2_fissures.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modèle TFLite créé: mobilenetv2_fissures.tflite")