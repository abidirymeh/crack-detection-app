# verifier_modele_utilise.py
import tensorflow as tf
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("🔍 VÉRIFICATION DU MODÈLE UTILISÉ DANS ANDROID")
print("=" * 60)

# 1. Charger le modèle
model = tf.keras.models.load_model('modele_materiaux_final.h5')
print(f"✅ Modèle chargé - Input shape: {model.input_shape}")

# 2. Charger les données
data = np.load('materiaux_prepared.npz')
X_test = data['X_test']
y_test = data['y_test']

# 3. Prédictions
y_pred = model.predict(X_test).flatten()

# 4. Métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RÉSULTATS :")
print(f"   MAE  : {mae:.2f} MPa")
print(f"   RMSE : {rmse:.2f} MPa")
print(f"   R²   : {r2:.4f}")

# 5. Vérifier la taille TFLite
import os
if os.path.exists('modele_materiaux_v1.tflite'):
    taille = os.path.getsize('modele_materiaux_v1.tflite') / 1024
    print(f"\n✅ TFLite trouvé: {taille:.0f} Ko")