# extraire_peinture_sdnet.py
import cv2
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import shutil

print("=" * 60)
print("🎨 EXTRACTION DATASET PEINTURE DEPUIS SDNET AVEC TRAIN/TEST")
print("=" * 60)

# Configuration
IMG_SIZE = 224
chemin_sdnet = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

# Dossiers de destination
train_ok = "../../dataset_peinture_augmente/train/ok"
train_abimee = "../../dataset_peinture_augmente/train/abimee"
test_ok = "../../dataset_peinture_augmente/test/ok"
test_abimee = "../../dataset_peinture_augmente/test/abimee"

for d in [train_ok, train_abimee, test_ok, test_abimee]:
    os.makedirs(d, exist_ok=True)

categories = ['Decks', 'Pavements', 'Walls']

# Listes pour stocker les images temporairement
all_ok = []
all_abimee = []

# Dossier temporaire
temp_dir = "temp_peinture"
os.makedirs(temp_dir, exist_ok=True)

for cat in categories:
    print(f"\n📁 {cat}:")
    
    # Images saines (peinture OK)
    path_ok = os.path.join(chemin_sdnet, cat, 'Non-cracked', '*.jpg')
    fichiers_ok = glob(path_ok)
    for i, f in enumerate(fichiers_ok[:500]):
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            nom = f"{cat}_ok_{i}.jpg"
            cv2.imwrite(os.path.join(temp_dir, nom), img)
            all_ok.append(nom)
    print(f"   OK: {len(fichiers_ok[:500])} images")
    
    # Images abîmées (peinture abîmée)
    path_abimee = os.path.join(chemin_sdnet, cat, 'Cracked', '*.jpg')
    fichiers_abimee = glob(path_abimee)
    for i, f in enumerate(fichiers_abimee[:500]):
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            nom = f"{cat}_abimee_{i}.jpg"
            cv2.imwrite(os.path.join(temp_dir, nom), img)
            all_abimee.append(nom)
    print(f"   Abîmée: {len(fichiers_abimee[:500])} images")

print(f"\n📊 Total: {len(all_ok)} OK, {len(all_abimee)} abîmées")

# Division train/test (80% train, 20% test)
train_ok_list, test_ok_list = train_test_split(all_ok, test_size=0.2, random_state=42)
train_abimee_list, test_abimee_list = train_test_split(all_abimee, test_size=0.2, random_state=42)

# Copier les fichiers
print("\n📁 Copie des fichiers...")

for nom in train_ok_list:
    shutil.move(os.path.join(temp_dir, nom), os.path.join(train_ok, nom))
for nom in test_ok_list:
    shutil.move(os.path.join(temp_dir, nom), os.path.join(test_ok, nom))
for nom in train_abimee_list:
    shutil.move(os.path.join(temp_dir, nom), os.path.join(train_abimee, nom))
for nom in test_abimee_list:
    shutil.move(os.path.join(temp_dir, nom), os.path.join(test_abimee, nom))

# Supprimer dossier temporaire
os.rmdir(temp_dir)

print(f"\n✅ Dataset peinture créé avec division train/test !")
print(f"   Train OK: {len(train_ok_list)} images")
print(f"   Train abîmée: {len(train_abimee_list)} images")
print(f"   Test OK: {len(test_ok_list)} images")
print(f"   Test abîmée: {len(test_abimee_list)} images")
print(f"   TOTAL: {len(all_ok) + len(all_abimee)} images")