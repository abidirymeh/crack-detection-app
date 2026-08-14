# creer_scaler_params.py
import tensorflow as tf
import joblib
import json

print("=" * 70)
print("📁 CRÉATION DU FICHIER SCALER_PARAMS.JSON")
print("=" * 70)

# 1. Charger le scaler
try:
    scaler = joblib.load('scaler_materiaux.pkl')
    print("✅ Scaler chargé")
    
    # 2. Extraire les paramètres
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    # 3. Sauvegarder en JSON
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print("✅ scaler_params.json créé")
    print(f"   Mean: {scaler_params['mean']}")
    print(f"   Scale: {scaler_params['scale']}")

except FileNotFoundError:
    print("❌ scaler_materiaux.pkl non trouvé")
    print("📁 Recherche dans le dossier analyse_materiaux...")
    
    # Chercher dans le dossier analyse_materiaux
    import os
    if os.path.exists('../analyse_materiaux/scaler_materiaux.pkl'):
        scaler = joblib.load('../analyse_materiaux/scaler_materiaux.pkl')
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print("✅ scaler_params.json créé depuis analyse_materiaux")
    else:
        print("❌ Fichier introuvable. Utilisation de valeurs par défaut")
        # Valeurs par défaut pour le test
        scaler_params = {
            'mean': [25.0, 60.0, 20.0, 5.0, 10.0],
            'scale': [20.0, 20.0, 15.0, 3.0, 15.0]
        }
        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print("✅ scaler_params.json créé avec valeurs par défaut")

print("\n✅ Terminé!")