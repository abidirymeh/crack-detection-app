# etape2_preparer.py
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("ÉTAPE 2: PRÉPARATION DES DONNÉES")
print("=" * 60)

IMG_SIZE = 224
chemin = "C:/Users/RIMEH/Desktop/PFA_ADEVA/03_dataset/sdnet"

def charger_images(chemin_base, max_par_classe=500):
    images = []
    labels = []
    
    # CORRIGÉ selon ta structure
    categories = ['Decks', 'Pavements', 'Walls'] 
    nom_cracked = 'Cracked'        
    nom_sain = 'Non-cracked'     
    
    total_images = 0
    
    for cat in categories:
        print(f"\n {cat}:")
        
        path_cracked = os.path.join(chemin_base, cat, nom_cracked, '*.jpg')
        fichiers = glob(path_cracked)
        print(f"   → Trouvé: {len(fichiers)} images avec fissures")
        
        for f in fichiers[:max_par_classe]:
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)
                total_images += 1
        
        path_sain = os.path.join(chemin_base, cat, nom_sain, '*.jpg')
        fichiers = glob(path_sain)
        print(f"   → Trouvé: {len(fichiers)} images sans fissures")
        
        for f in fichiers[:max_par_classe]:
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)
                total_images += 1
    
    print(f"\n Total images chargées: {total_images}")
    return np.array(images), np.array(labels)

print("\n Chargement en cours...")
X, y = charger_images(chemin, max_par_classe=500)

print(f"\n Résultat final:")
print(f"   - Total: {len(X)} images")
print(f"   - Avec fissures: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
print(f"   - Sans fissures: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Split:")
print(f"   - Train: {len(X_train)} images")
print(f"   - Test: {len(X_test)} images")

X_train = X_train / 255.0
X_test = X_test / 255.0

np.savez('sdnet_prepared.npz', 
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)

print("\n Données sauvegardées dans 'sdnet_prepared.npz'")
print("\n ÉTAPE 2 TERMINÉE! Prêt pour l'ÉTAPE 3")