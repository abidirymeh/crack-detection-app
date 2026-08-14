import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("=" * 70)
print("ÉTAPE 7.3: MODÈLE DE PRÉDICTION MATÉRIAUX")
print("=" * 70)

print("\n📥 Chargement des données...")
data = np.load('materiaux_prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print(f" Données chargées: {len(X_train)} train, {len(X_test)} test")

print("\n Création du modèle de régression...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Sortie pour régression
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'meilleur_modele_materiaux.h5',
        monitor='val_mae',
        save_best_only=True,
        verbose=1
    )
]

print("\n Entraînement en cours...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n ÉVALUATION FINALE")
print("=" * 50)

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n Métriques:")
print(f"   - MAE (Mean Absolute Error): {mae:.2f} MPa")
print(f"   - MSE (Mean Squared Error): {mse:.2f}")
print(f"   - RMSE (Root Mean Squared Error): {rmse:.2f} MPa")
print(f"   - R² Score: {r2:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Train')
plt.plot(history.history['val_mae'], label='Validation')
plt.title('MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# Prédictions vs Réalité
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valeurs réelles (MPa)')
plt.ylabel('Prédictions (MPa)')
plt.title(f'Prédictions vs Réalité\nR² = {r2:.3f}')
plt.grid(True)

plt.tight_layout()
plt.savefig('resultats_materiaux.png')
print(f"\n Graphiques sauvegardés: resultats_materiaux.png")

# 7. SAUVEGARDER
model.save('modele_materiaux_final.h5')
print(f"\n Modèle final sauvegardé: modele_materiaux_final.h5")

print("\n ÉTAPE 7.3 TERMINÉE!")