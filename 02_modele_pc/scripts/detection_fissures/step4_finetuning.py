# step4_finetuning.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("=" * 60)
print("ÉTAPE 4: FINE-TUNING DE MOBILENETV3-SMALL")
print("=" * 60)

# 1. CHARGER LES DONNÉES
print("\n Chargement des données...")
data = np.load('sdnet_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print(f" Données chargées:")
print(f"   - Train: {len(X_train)} images")
print(f"   - Test: {len(X_test)} images")

# 2. CRÉER LE MODÈLE
print("\n Création du modèle...")

# Modèle de base (gelé pour commencer)
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False  # On gèle d'abord

# Ajouter notre tête de classification
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)  # Pour éviter l'overfitting
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(f" Modèle créé: {model.count_params():,} paramètres")

# 3. CALLBACKS (pour sauvegarder le meilleur modèle)
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'meilleur_modele.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 4. PHASE 1: Entraîner seulement la tête
print("\n" + "=" * 60)
print("PHASE 1: Entraînement de la tête (couches ajoutées)")
print("=" * 60)

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 5. PHASE 2: Fine-tuning complet (optionnel)
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning complet (toutes les couches)")
print("=" * 60)

base_model.trainable = True  # On dégèle le modèle de base
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # LR plus petit
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. ÉVALUATION FINALE
print("\n" + "=" * 60)
print(" ÉVALUATION FINALE")
print("=" * 60)

# Prédictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Métriques
print("\n Rapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Sain', 'Fissure']))

print("\n Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 7. SAUVEGARDER LE MODÈLE FINAL
model.save('mobilenetv3_sdnet_final.h5')
print("\n Modèle final sauvegardé: mobilenetv3_sdnet_final.h5")

# 8. VISUALISER L'ENTRAÎNEMENT
print("\n Génération des courbes d'entraînement...")

# Combiner les historiques
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('courbes_entrainement.png')
plt.show()

print("\n ENTRAÎNEMENT TERMINÉ!")
print("   - Meilleur modèle: meilleur_modele.h5")
print("   - Modèle final: mobilenetv3_sdnet_final.h5")
print("   - Courbes: courbes_entrainement.png")