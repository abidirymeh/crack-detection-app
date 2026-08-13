# train_enriched_fixed.py
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("=" * 70)
print("🚀 ENTRAÎNEMENT SUR DATASET ENRICHI (SDNET + CCIC + VOS IMAGES)")
print("=" * 70)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_CCIC_IMAGES = 5000  # Limiter CCIC pour éviter erreur mémoire

# 1. Fonction pour charger des images avec limite
def charger_images_dossier(dossier, label, max_images=None):
    images = []
    labels = []
    
    if not os.path.exists(dossier):
        print(f"   ⚠️ Dossier non trouvé: {dossier}")
        return images, labels
    
    compteur = 0
    for fichier in os.listdir(dossier):
        if max_images and compteur >= max_images:
            break
            
        if fichier.lower().endswith(('.jpg', '.png', '.jpeg')):
            chemin = os.path.join(dossier, fichier)
            img = cv2.imread(chemin)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0  # Conversion en float32 (économie mémoire)
                images.append(img)
                labels.append(label)
                compteur += 1
                
                if compteur % 1000 == 0:
                    print(f"      → {compteur} images chargées")
    
    return np.array(images, dtype=np.float32), np.array(labels)

# 2. Charger SDNET
print("\n📁 Chargement de SDNET...")
data_sdnet = np.load('sdnet_prepared.npz')
X_train_sdnet = data_sdnet['X_train'].astype(np.float32)
y_train_sdnet = data_sdnet['y_train']
X_test_sdnet = data_sdnet['X_test'].astype(np.float32)
y_test_sdnet = data_sdnet['y_test']
print(f"   SDNET train: {len(X_train_sdnet)} images")
print(f"   SDNET test: {len(X_test_sdnet)} images")

# 3. Charger CCIC (limité)
print("\n📁 Chargement de CCIC (limité)...")
X_ccic_sain, y_ccic_sain = charger_images_dossier('../../../03_dataset/ccic/Negative', 0, MAX_CCIC_IMAGES // 2)
X_ccic_fissure, y_ccic_fissure = charger_images_dossier('../../../03_dataset/ccic/Positive', 1, MAX_CCIC_IMAGES // 2)

X_ccic = np.concatenate([X_ccic_sain, X_ccic_fissure]) if len(X_ccic_sain) > 0 else np.array([])
y_ccic = np.concatenate([y_ccic_sain, y_ccic_fissure]) if len(y_ccic_sain) > 0 else np.array([])
print(f"   CCIC chargé: {len(X_ccic)} images")

# 4. Charger vos images augmentées
print("\n📁 Chargement de vos images augmentées...")
X_vos_sain, _ = charger_images_dossier('augmented/sain', 0)
X_vos_fissure, _ = charger_images_dossier('augmented/fissure', 1)

X_vos = np.concatenate([X_vos_sain, X_vos_fissure]) if len(X_vos_sain) > 0 else np.array([])
y_vos = np.concatenate([np.zeros(len(X_vos_sain)), np.ones(len(X_vos_fissure))]) if len(X_vos_sain) > 0 else np.array([])
print(f"   Vos images augmentées: {len(X_vos)} images")

# 5. COMBINER
print("\n🔗 Fusion des datasets...")
X_train_combined = X_train_sdnet
y_train_combined = y_train_sdnet

if len(X_ccic) > 0:
    X_train_combined = np.concatenate([X_train_combined, X_ccic])
    y_train_combined = np.concatenate([y_train_combined, y_ccic])

if len(X_vos) > 0:
    X_train_combined = np.concatenate([X_train_combined, X_vos])
    y_train_combined = np.concatenate([y_train_combined, y_vos])

X_test_final = X_test_sdnet
y_test_final = y_test_sdnet

print(f"\n   Dataset d'entraînement final: {len(X_train_combined)} images")
print(f"   Dataset de test final (SDNET): {len(X_test_final)} images")

# 6. Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# 7. Modèle MobileNetV2
print("\n🏗️ Construction du modèle...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("✅ Modèle créé")

# 8. Entraînement
print("\n🔥 ENTRAÎNEMENT - 20 ÉPOQUES")
history = model.fit(
    datagen.flow(X_train_combined, y_train_combined, batch_size=BATCH_SIZE),
    validation_data=(X_test_final, y_test_final),
    epochs=20,
    verbose=1
)

# 9. Évaluation finale
print("\n📊 ÉVALUATION FINALE SUR SDNET")
y_pred_proba = model.predict(X_test_final)
y_pred = (y_pred_proba > 0.5).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

prec = precision_score(y_test_final, y_pred)
rec = recall_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
acc = accuracy_score(y_test_final, y_pred)

print("\n" + "=" * 70)
print("📊 RÉSULTATS FINAUX")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCES APRÈS ENRICHISSEMENT                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   📊 Composition de l'entraînement :                                    │
│      • SDNET train: {len(X_train_sdnet)} images                                  │
│      • CCIC: {len(X_ccic)} images                                              │
│      • Vos images: {len(X_vos)} images                                           │
│      • TOTAL: {len(X_train_combined)} images                                      │
│                                                                          │
│   📊 Résultats sur SDNET test (600 images) :                            │
│                                                                          │
│      🎯 PRÉCISION : {prec:.2%} ({prec*100:.1f}%)                                         │
│      🔍 RECALL    : {rec:.2%} ({rec*100:.1f}%)                                         │
│      ⭐ F1-SCORE  : {f1:.2%} ({f1*100:.1f}%)                                         │
│      📈 ACCURACY  : {acc:.2%} ({acc*100:.1f}%)                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 10. Sauvegarde
model.save('modele_enriched.h5')

# Conversion TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('modele_enriched.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n✅ Modèle sauvegardé: modele_enriched.tflite")