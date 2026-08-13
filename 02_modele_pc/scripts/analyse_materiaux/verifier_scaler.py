# verifier_scaler.py
import joblib
import numpy as np

# Charger le scaler
scaler = joblib.load('scaler_materiaux.pkl')

print("Moyennes :", scaler.mean_)
print("Écarts-types :", scaler.scale_)

