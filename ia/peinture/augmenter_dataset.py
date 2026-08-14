from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
import os

print("=" * 60)
print("🎨 AUGMENTATION DU DATASET PEINTURE")
print("=" * 60)

IMG_SIZE = 224
NB_IMAGES_PAR_ORIGINALE = 30  # 15 variations par image

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3]
)

source_dirs = {
    'ok': r'C:\Users\RIMEH\Desktop\PFA_ADEVA\02_modele_pc\dataset_peinture\train\ok',
    'abimee': r'C:\Users\RIMEH\Desktop\PFA_ADEVA\02_modele_pc\dataset_peinture\train\abimee'
}

dest_dirs = {
    'ok': r'C:\Users\RIMEH\Desktop\PFA_ADEVA\02_modele_pc\dataset_peinture_augmente\train\ok',
    'abimee': r'C:\Users\RIMEH\Desktop\PFA_ADEVA\02_modele_pc\dataset_peinture_augmente\train\abimee'
}

for classe, path in source_dirs.items():
    if not os.path.exists(path):
        print(f"❌ Dossier non trouvé: {path}")
        print(f"   Crée le dossier et mets tes images {classe} dedans")
        exit()

for dest in dest_dirs.values():
    os.makedirs(dest, exist_ok=True)

for classe, src_path in source_dirs.items():
    images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\n📁 {classe}: {len(images)} images originales")
    
    if len(images) == 0:
        print(f"   ⚠️ Aucune image trouvée dans {src_path}")
        continue
    
    for img_file in images:
        img_path = os.path.join(src_path, img_file)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Sauvegarder l'originale
        dest_path = os.path.join(dest_dirs[classe], f"original_{img_file}")
        array_to_img(img_array[0]).save(dest_path)
        
        # Générer les variations
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=dest_dirs[classe],
                                   save_prefix=f"aug_{img_file.split('.')[0]}", save_format='jpg'):
            i += 1
            if i >= NB_IMAGES_PAR_ORIGINALE:
                break
        
        print(f"   ✅ {img_file}: {NB_IMAGES_PAR_ORIGINALE} variations générées")

print("\n" + "=" * 60)
print("✅ Dataset augmenté créé !")
print(f"   Dossier: dataset_peinture_augmente/train/")
print("=" * 60)