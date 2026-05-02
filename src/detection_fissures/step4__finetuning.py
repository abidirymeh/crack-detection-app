# step4_finetuning.py - VERSION MULTI-SORTIES (FISSURE + PEINTURE)
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("=" * 60)
print("ÉTAPE 4: FINE-TUNING MULTI-SORTIES")
print("MobileNetV3-Small : FISSURE + PEINTURE")
print("=" * 60)

# 1. CHARGER LES DONNÉES (fissures)
print("\n📥 Chargement des données fissures...")
data = np.load('sdnet_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_fissure_train, y_fissure_test = data['y_train'], data['y_test']

print(f" Données chargées:")
print(f"   - Train: {len(X_train)} images")
print(f"   - Test: {len(X_test)} images")
print(f"   - Fissures: {np.sum(y_fissure_train)} train, {np.sum(y_fissure_test)} test")

# 2. CRÉER DES LABELS PEINTURE (SIMULÉS POUR L'INSTANT)
# ⚠️ À remplacer par tes vrais labels peinture quand tu auras collecté les images
print("\n🎨 Création des labels peinture (simulés)...")
np.random.seed(42)
y_peinture_train = np.random.randint(0, 2, len(y_fissure_train))
y_peinture_test = np.random.randint(0, 2, len(y_fissure_test))

print(f"   - Peinture abîmée: {np.sum(y_peinture_train)} train, {np.sum(y_peinture_test)} test")

# 3. CRÉER LE MODÈLE MULTI-SORTIES
print("\n🏗️ Création du modèle multi-sorties...")

# Modèle de base
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
base_model.trainable = False  # On gèle d'abord

# Architecture multi-sorties
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)

# DEUX SORTIES :
# - Sortie 1 : Fissure (0=sain, 1=fissure)
# - Sortie 2 : Peinture (0=bonne, 1=abîmée)
output_fissure = tf.keras.layers.Dense(1, activation='sigmoid', name='fissure')(x)
output_peinture = tf.keras.layers.Dense(1, activation='sigmoid', name='peinture')(x)

model = tf.keras.Model(inputs, outputs=[output_fissure, output_peinture])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'fissure': 'binary_crossentropy',
        'peinture': 'binary_crossentropy'
    },
    metrics={
        'fissure': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        'peinture': ['accuracy']
    }
)

print(f" Modèle créé: {model.count_params():,} paramètres")
print(f" Sorties: fissure, peinture")

# 4. CALLBACKS
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
        'meilleur_modele_multitask.h5',
        monitor='val_fissure_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 5. PHASE 1: Entraîner seulement la tête
print("\n" + "=" * 60)
print("PHASE 1: Entraînement de la tête (couches ajoutées)")
print("=" * 60)

history1 = model.fit(
    X_train, 
    {'fissure': y_fissure_train, 'peinture': y_peinture_train},
    validation_data=(X_test, {'fissure': y_fissure_test, 'peinture': y_peinture_test}),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. PHASE 2: Fine-tuning complet
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning complet (toutes les couches)")
print("=" * 60)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={
        'fissure': 'binary_crossentropy',
        'peinture': 'binary_crossentropy'
    },
    metrics={
        'fissure': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        'peinture': ['accuracy']
    }
)

history2 = model.fit(
    X_train,
    {'fissure': y_fissure_train, 'peinture': y_peinture_train},
    validation_data=(X_test, {'fissure': y_fissure_test, 'peinture': y_peinture_test}),
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7. ÉVALUATION FINALE
print("\n" + "=" * 60)
print("📊 ÉVALUATION FINALE")
print("=" * 60)

# Prédictions
predictions = model.predict(X_test)
y_fissure_pred = (predictions[0] > 0.5).astype(int).flatten()
y_peinture_pred = (predictions[1] > 0.5).astype(int).flatten()

print("\n🔍 RÉSULTATS FISSURES:")
print(classification_report(y_fissure_test, y_fissure_pred, target_names=['Sain', 'Fissure']))

print("\n🎨 RÉSULTATS PEINTURE:")
print(classification_report(y_peinture_test, y_peinture_pred, target_names=['Bonne', 'Abîmée']))

# 8. SAUVEGARDER LE MODÈLE FINAL
model.save('mobilenetv3_multitask_final.h5')
print("\n💾 Modèle final sauvegardé: mobilenetv3_multitask_final.h5")

# 9. CONVERSION TFLite
print("\n🔄 Conversion en TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open('mobilenetv3_multitask.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ Modèle TFLite créé: mobilenetv3_multitask.tflite")
print(f"   Taille: {len(tflite_model) / (1024*1024):.2f} Mo")

# 10. VISUALISATION
print("\n📈 Génération des courbes d'entraînement...")

plt.figure(figsize=(15, 10))

# Accuracy fissure
plt.subplot(2, 2, 1)
acc_f = history1.history['fissure_accuracy'] + history2.history['fissure_accuracy']
val_acc_f = history1.history['val_fissure_accuracy'] + history2.history['val_fissure_accuracy']
plt.plot(acc_f, label='Train')
plt.plot(val_acc_f, label='Validation')
plt.title('Accuracy - Fissure')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Accuracy peinture
plt.subplot(2, 2, 2)
acc_p = history1.history['peinture_accuracy'] + history2.history['peinture_accuracy']
val_acc_p = history1.history['val_peinture_accuracy'] + history2.history['val_peinture_accuracy']
plt.plot(acc_p, label='Train')
plt.plot(val_acc_p, label='Validation')
plt.title('Accuracy - Peinture')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss globale
plt.subplot(2, 2, 3)
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
plt.plot(loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.title('Loss totale')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Recall fissure
plt.subplot(2, 2, 4)
recall_f = history1.history['fissure_recall'] + history2.history['fissure_recall']
val_recall_f = history1.history['val_fissure_recall'] + history2.history['val_fissure_recall']
plt.plot(recall_f, label='Train')
plt.plot(val_recall_f, label='Validation')
plt.title('Recall - Fissure')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('courbes_entrainement_multitask.png')
plt.show()

print("\n" + "=" * 60)
print("✅ ENTRAÎNEMENT MULTI-TÂCHES TERMINÉ !")
print("=" * 60)
print("\n📁 Fichiers générés:")
print("   - meilleur_modele_multitask.h5")
print("   - mobilenetv3_multitask_final.h5")
print("   - mobilenetv3_multitask.tflite (pour Android)")
print("   - courbes_entrainement_multitask.png")