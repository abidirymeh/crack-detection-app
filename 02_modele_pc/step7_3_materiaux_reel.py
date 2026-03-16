# step7_3_materiaux_reel.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("ENTRAÎNEMENT SUR UCI CONCRETE DATASET")
print("=" * 70)

# 1. CHARGER LES DONNÉES PRÉPARÉES
print("\n📥 Chargement des données...")
data = np.load('concrete_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print(f"✅ Données chargées: {len(X_train)} train, {len(X_test)} test")

# 2. CRÉER LE MODÈLE
print("\n🤖 Création du modèle...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# 3. CALLBACKS
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'meilleur_modele_concrete.h5',
        monitor='val_mae',
        save_best_only=True,
        verbose=1
    )
]

# 4. ENTRAÎNEMENT
print("\n⏳ Entraînement en cours...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 5. ÉVALUATION
print("\n📊 ÉVALUATION FINALE")
print("=" * 50)

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Métriques:")
print(f"   - MAE: {mae:.2f} MPa")
print(f"   - RMSE: {rmse:.2f} MPa")
print(f"   - R² Score: {r2:.4f}")

# 6. VISUALISATION
plt.figure(figsize=(15, 5))

# Courbes d'apprentissage
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Train')
plt.plot(history.history['val_mae'], label='Validation')
plt.title('MAE')
plt.xlabel('Epochs')
plt.legend()

# Prédictions vs Réalité
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valeurs réelles (MPa)')
plt.ylabel('Prédictions (MPa)')
plt.title(f'Prédictions vs Réalité\nR² = {r2:.3f}')

plt.tight_layout()
plt.savefig('resultats_concrete.png')
print(f"\n✅ Graphiques: resultats_concrete.png")

# 7. SAUVEGARDER
model.save('modele_concrete_final.h5')
print(f"\n💾 Modèle final: modele_concrete_final.h5")

print("\n✅ ENTRAÎNEMENT TERMINÉ!")