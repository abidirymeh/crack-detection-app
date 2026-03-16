# step3_test_modele.py
import tensorflow as tf
import numpy as np

print("=" * 60)
print("ÉTAPE 3: TEST RAPIDE DE MOBILENETV3-SMALL")
print("=" * 60)

# Charger les données préparées
print("\n📥 Chargement des données...")
data = np.load('sdnet_prepared.npz')
X_test = data['X_test']
y_test = data['y_test']

print(f"✅ Données chargées: {len(X_test)} images de test")

# Charger MobileNetV3-Small (pré-entraîné sur ImageNet)
print("\n🤖 Chargement de MobileNetV3-Small...")
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

print(f"✅ Modèle de base chargé")
print(f"   - Paramètres: {base_model.count_params():,}")

# Ajouter notre tête de classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n✅ Modèle complet créé")
print(f"   - Total paramètres: {model.count_params():,}")

# Test sur un petit batch
print("\n⚡ Test sur 32 images...")
batch_x = X_test[:32]
batch_y = y_test[:32]
loss, acc = model.evaluate(batch_x, batch_y, verbose=0)
print(f"   → Accuracy: {acc:.4f} (sur 32 images)")

print("\n✅ TEST RÉUSSI!")
print("   Le modèle fonctionne correctement.")
print("   Prêt pour l'ÉTAPE 4 (fine-tuning)!")