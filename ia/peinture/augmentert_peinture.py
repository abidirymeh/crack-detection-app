import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import precision_score, recall_score, f1_score

print("=" * 60)
print("🎨 AUGMENTATION + TEST SUR 23 PHOTOS PERSONNELLES")
print("=" * 60)

interpreter = tf.lite.Interpreter('modele_peinture_v2.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img):
    inp = np.expand_dims(img, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def augmenter_image(img):
    versions = []
    h, w = img.shape[:2]
    
    versions.append(img)
    
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1)
    versions.append(cv2.warpAffine(img, M, (w, h)))
    M = cv2.getRotationMatrix2D((w/2, h/2), -10, 1)
    versions.append(cv2.warpAffine(img, M, (w, h)))
    
    versions.append(cv2.flip(img, 1))
    
    versions.append(cv2.convertScaleAbs(img, alpha=1.3, beta=15))
    versions.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-15))
    
    zoom = cv2.resize(img[20:-20, 20:-20], (w, h))
    versions.append(zoom)
    
    versions.append(cv2.GaussianBlur(img, (3, 3), 0))
    
    return versions

dossier_test = r"C:\Users\RIMEH\Desktop\testPeinture"

vraies_classes = {
    'ab1.jpg': 0, 'ab10.jpg': 0, 'ab11.jpg': 0, 'ab12.jpg': 0, 'ab13.jpg': 0,
    'ab2.jpg': 0, 'ab3.jpg': 0, 'ab4.jpg': 0, 'ab5.jpg': 0, 'ab6.jpg': 0,
    'ab7.jpg': 0, 'ab8.jpg': 0, 'ab9.jpg': 0,
    'im1.jpg': 1, 'im10.jpg': 1, 'im2.jpg': 1, 'im3.jpg': 1, 'im4.jpg': 1,
    'im5.jpg': 1, 'im6.jpg': 1, 'im7.jpg': 1, 'im8.jpg': 1, 'im9.jpg': 1
}

images_originales = []
labels_originales = []

for nom, label in vraies_classes.items():
    img = cv2.imread(os.path.join(dossier_test, nom))
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_originales.append(img)
        labels_originales.append(label)

print(f"📊 {len(images_originales)} images originales chargées")

print("\n🔄 Augmentation et test...")

X_test_augmente = []
y_test_augmente = []

for i, img in enumerate(images_originales):
    versions = augmenter_image(img)
    for v in versions:
        X_test_augmente.append(v)
        y_test_augmente.append(labels_originales[i])

print(f"   → {len(X_test_augmente)} images après augmentation")

probas = []
for img in X_test_augmente:
    img_norm = img / 255.0
    proba = predict(img_norm)
    probas.append(proba)

probas = np.array(probas)
y_test_augmente = np.array(y_test_augmente)

print("\n" + "=" * 60)
print("📊 RÉSULTATS SUR IMAGES AUGMENTÉES")
print("=" * 60)

print(f"\n{'Seuil':>6} {'Précision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 40)

for seuil in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    y_pred = (probas > seuil).astype(int)
    prec = precision_score(y_test_augmente, y_pred)
    rec = recall_score(y_test_augmente, y_pred)
    f1 = f1_score(y_test_augmente, y_pred)
    print(f"{seuil:.2f}   {prec:>9.2%} {rec:>9.2%} {f1:>9.2%}")

print("\n" + "=" * 60)
print("📊 COMPARAISON AVEC IMAGES ORIGINALES SEULES")
print("=" * 60)

probas_orig = []
for img in images_originales:
    img_norm = img / 255.0
    proba = predict(img_norm)
    probas_orig.append(proba)

print(f"\n{'Seuil':>6} {'Précision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 40)

for seuil in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    y_pred = (np.array(probas_orig) > seuil).astype(int)
    prec = precision_score(labels_originales, y_pred)
    rec = recall_score(labels_originales, y_pred)
    f1 = f1_score(labels_originales, y_pred)
    print(f"{seuil:.2f}   {prec:>9.2%} {rec:>9.2%} {f1:>9.2%}")

print("\n" + "=" * 60)
print("🏆 MEILLEUR SEUIL RECOMMANDÉ")
print("=" * 60)

best_f1 = 0
best_seuil = 0.65
for seuil in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    y_pred = (probas > seuil).astype(int)
    f1 = f1_score(y_test_augmente, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_seuil = seuil
        best_prec = precision_score(y_test_augmente, y_pred)
        best_rec = recall_score(y_test_augmente, y_pred)

print(f"\n🔧 Seuil optimal: {best_seuil:.2f}")
print(f"   → Précision: {best_prec:.2%}")
print(f"   → Recall: {best_rec:.2%}")
print(f"   → F1: {best_f1:.2%}")

print("\n✅ Test terminé !")