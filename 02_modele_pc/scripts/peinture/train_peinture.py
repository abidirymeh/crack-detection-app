# train_peinture.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

print("=" * 60)
print("🎨 ENTRAÎNEMENT MODÈLE PEINTURE (DATASET AUGMENTÉ)")
print("=" * 60)

# 1. Dossiers
train_path = '../../dataset_peinture_augmente/train'
test_path = '../../dataset_peinture_augmente/test'  # Garde tes images de test originales

# Vérifier que les dossiers existent
if not os.path.exists(train_path):
    print(f"❌ Dossier train non trouvé: {train_path}")
    print("   Exécute d'abord augmenter_dataset.py")
    exit()

# 2. Paramètres
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

# 3. Data augmentation (légère pour l'entraînement)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 4. Charger les images
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Vérifier si le dossier test existe
if os.path.exists(test_path):
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    validation_data = test_generator
else:
    print(f"⚠️ Dossier test non trouvé, utilisation de 20% des données train pour validation")
    validation_data = None
    # Utiliser une partie des données train pour validation
    validation_split = 0.2

print(f"\n📊 Classes trouvées: {train_generator.class_indices}")
print(f"   Train: {train_generator.samples} images")

# 5. Modèle MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n✅ Modèle MobileNetV2 créé")

# 6. Entraînement
print(f"\n🔥 Entraînement sur {EPOCHS} époques...")

if validation_data:
    history = model.fit(
        train_generator,
        validation_data=validation_data,
        epochs=EPOCHS,
        verbose=1
    )
else:
    history = model.fit(
        train_generator,
        validation_split=validation_split,
        epochs=EPOCHS,
        verbose=1
    )

# 7. Évaluation finale
print("\n📊 ÉVALUATION FINALE:")
if validation_data:
    loss, acc = model.evaluate(test_generator)
    print(f"   Précision sur test: {acc*100:.1f}%")

# 8. Sauvegarde
model.save('modele_peinture_v2.h5')

# 9. Conversion TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('modele_peinture_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"\n✅ Modèle créé: modele_peinture_v2.tflite")
print(f"   Taille: {len(tflite_model) / (1024*1024):.2f} Mo")

# 10. Afficher les courbes
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig('courbes_peinture_v2.png')
plt.show()

print("\n✅ Entraînement terminé !")