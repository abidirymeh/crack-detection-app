# etape1_explorer.py
import os
from glob import glob

print("=" * 60)
print("ÉTAPE 1: EXPLORATION DE SDNET2018")
print("=" * 60)

chemin = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

if not os.path.exists(chemin):
    print(f"\n Dossier introuvable: {chemin}")
    print(" Vérifie le chemin exact de ton dossier SDNET2018")
    exit()

print(f"\n Dossier trouvé: {chemin}")

# Afficher la structure
print("\n STRUCTURE DU DATASET:")
for dossier in os.listdir(chemin):
    dossier_path = os.path.join(chemin, dossier)
    if os.path.isdir(dossier_path):
        print(f"\n {dossier}/")

        
        # Chercher les sous-dossiers (cracked, non-cracked)
        for sous_dossier in os.listdir(dossier_path):
            sous_path = os.path.join(dossier_path, sous_dossier)
            if os.path.isdir(sous_path):
                nb_images = len(glob(os.path.join(sous_path, "*.jpg")))
                print(f"   └─ {sous_dossier}/: {nb_images} images")