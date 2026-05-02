# train_peinture_agca.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

print("=" * 60)
print("🎨 ENTRAÎNEMENT MODÈLE PEINTURE AVEC AGCA")
print("=" * 60)

# 1. Dossiers (inchangés)
train_path = '../../dataset_peinture_augmente/train'
test_path = '../../dataset_peinture_augmente/test'

# 2. Paramètres
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

# 3. Data augmentation (inchangée)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 4. Charger les images (inchangé)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"\n📊 Classes: {train_generator.class_indices}")
print(f"   Train: {train_generator.samples} images")
print(f"   Test: {test_generator.samples} images")

# 5. 🔥 NOUVEAU : MODULE AGCA (Adaptive Graph Channel Attention)
def agca_module(x, reduction_ratio=16):
    """
    Adaptive Graph Channel Attention Module
    """
    # Sauvegarder les dimensions
    channels = x.shape[-1]
    
    # Global Average Pooling
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers pour l'attention
    fc1 = tf.keras.layers.Dense(channels // reduction_ratio, activation='relu')(gap)
    fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')(fc1)
    
    # Reshape pour l'attention
    attention = tf.keras.layers.Reshape((1, 1, channels))(fc2)
    
    # Appliquer l'attention (multiplication élément par élément)
    return tf.keras.layers.Multiply()([x, attention])

# 6. 🔥 NOUVEAU : MODÈLE MOBILENETV2 AVEC AGCA
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet',
    include_top=False,
    pooling=None  # Pas de pooling pour garder les dimensions
)
base_model.trainable = False  # Pas de fine-tuning

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)

# 🔥 AJOUT DU MODULE AGCA
x = agca_module(x, reduction_ratio=16)

# Pooling global
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Couches de classification
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(f"\n✅ Modèle MobileNetV2 + AGCA créé")
print(f"   Total paramètres: {model.count_params():,}")

# 7. Entraînement (inchangé)
print(f"\n🔥 Entraînement sur {EPOCHS} époques...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    verbose=1
)

# 8. Évaluation finale
print("\n📊 ÉVALUATION FINALE:")
loss, acc, prec, rec = model.evaluate(test_generator)
print(f"   Précision sur test: {acc*100:.1f}%")
print(f"   Précision (Precision): {prec*100:.1f}%")
print(f"   Recall: {rec*100:.1f}%")

# 9. Sauvegarde
model.save('modele_peinture_agca.h5')

# 10. Conversion TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('modele_peinture_agca.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Modèle créé: modele_peinture_agca.tflite")
print(f"   Taille: {len(tflite_model) / (1024*1024):.2f} Mo")

# 11. Afficher les courbes
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig('courbes_peinture_agca.png')
plt.show()

print("\n✅ Entraînement terminé !")