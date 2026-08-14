# PFA_ADEVA — Détection de fissures & Diagnostic bâtiment

## À propos

Ce projet vise à faciliter le diagnostic de l'état d'un bâtiment (façades, murs) directement depuis un smartphone. À partir d'une simple photo et de quelques informations sur les conditions du bâtiment, l'application détecte automatiquement la présence de **fissures**, évalue l'**état de la peinture** et estime la **résistance du matériau** (béton), grâce à des modèles de Deep Learning embarqués (TensorFlow Lite). Le diagnostic peut être affiné par une analyse complémentaire via un **LLM (Groq)** et archivé de façon infalsifiable sur la **blockchain Hedera**, dans un objectif de traçabilité pour la maintenance du bâtiment .

## Démonstration

https://github.com/user-attachments/assets/0e760895-a8b4-46db-8153-72c51cff1c92
https://github.com/user-attachments/assets/faf0218d-afd5-4a6b-8c94-fd0988d1d70f
https://github.com/user-attachments/assets/3c3398fc-217d-461e-bd21-2cb2fe2957eb
https://github.com/user-attachments/assets/c7137b42-7a23-42f9-beb2-167002d84e0e
https://github.com/user-attachments/assets/1765306c-17e6-4762-b6f0-d0f9b21aa065

## Table des matières

- 🪧 [À propos](#à-propos)
- 📦 [Prérequis](#prérequis)
- 🚀 [Installation](#installation)
- 🛠️ [Utilisation](#utilisation)
- 🤝 [Contribution](#contribution)
- 🏗️ [Construit avec](#construit-avec)
- 📚 [Documentation](#documentation)
- 🏷️ [Gestion des versions](#gestion-des-versions)
- 📝 [Licence](#licence)
- 👤 [Auteur](#auteur)

## Prérequis

Le projet est composé de deux parties (`ia/` et `app_mobile/`), chacune avec ses propres prérequis.

**Partie `ia/` (entraînement des modèles) :**

- [Python 3.9+](https://www.python.org/downloads/) — langage utilisé pour tous les scripts de préparation et d'entraînement
- [TensorFlow](https://www.tensorflow.org/install) — construction et entraînement des réseaux de neurones (Keras)
- [scikit-learn](https://scikit-learn.org/stable/install.html) — split train/test, normalisation (`StandardScaler`), métriques
- [OpenCV (opencv-python)](https://pypi.org/project/opencv-python/) — lecture, redimensionnement et prétraitement des images
- [NumPy](https://numpy.org/install/) / [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) — manipulation des données
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) / [Seaborn](https://seaborn.pydata.org/installing.html) — visualisation (distributions, corrélations, courbes d'entraînement)
- [joblib](https://joblib.readthedocs.io/en/stable/) — sauvegarde/chargement du scaler
- Datasets : [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) (images de fissures), [CCIC](https://data.mendeley.com/datasets/5y9wdsg2zt/2) (Concrete Crack Images for Classification), [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

**Partie `app_mobile/` (application Android) :**

- [Android Studio](https://developer.android.com/studio) — IDE de développement
- [JDK 11+](https://developer.android.com/build/jdks) — compilation du projet Android
- [Android SDK](https://developer.android.com/tools) (niveau d'API à préciser selon `build.gradle`) — outils de build Android
- Un appareil Android physique ou un émulateur disposant d'une **caméra**, la reconnaissance des fissures nécessitant une prise de vue en direct
- Accès réseau à un backend exposant :
  - une API d'analyse LLM (`/api/v1/analyze?model=groq`) — voir [documentation Groq](https://console.groq.com/docs)
  - une API d'enregistrement blockchain (`/api/v1/blockchain/record`) — voir [documentation Hedera](https://docs.hedera.com/)

## Installation

**1. Cloner le dépôt**

```bash
git clone https://github.com/abidirymeh/crack-detection-app.git
cd crack-detection-app
```

**2. Installer l'environnement Python (`ia/`)**

```bash
cd ia
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
```

**3. Récupérer et placer les datasets**

Télécharger SDNET2018, CCIC et UCI Concrete puis adapter les chemins d'accès (actuellement en dur, ex. `C:/Users/.../03_dataset/`) en tête des scripts concernés.

**4. Ouvrir le projet Android (`app_mobile/`)**

```bash
cd ../app_mobile
```

Ouvrir le dossier dans Android Studio, laisser Gradle synchroniser les dépendances (CameraX, TensorFlow Lite, OkHttp), puis renseigner l'URL du backend dans `MainActivity.java` (constante `API_BASE`).

**5. Copier les modèles entraînés dans l'application**

Une fois les modèles `.tflite` générés côté `ia/`, les placer dans :

```bash
cp modele_enriched.tflite modele_concrete.tflite modele_peinture_agca.tflite \
   app_mobile/app/src/main/assets/
```

## Utilisation

### Entraîner / régénérer les modèles (`ia/`)

```bash
# 1. Détection de fissures
python step1verifier_Structure_dataset.py
python step2preparerdonnee.py
python train.py                     # → modele_enriched.tflite

# 2. Résistance des matériaux (dataset réel recommandé)
python 2b_preparer_uci_concrete.py
python 3b_entrainer_uci_concrete.py
python 6_convertir_tflite.py        # → modele_concrete.tflite
python scaler_params.py             # → scaler_params.json

# 3. État de la peinture
python aug.py
python augmenter_dataset.py
python augmentert_peinture.py       # calibration du seuil de décision

# 4. Test d'intégration global (fissures + matériaux)
python 5_integration_fissures.py
```

### Lancer l'application en mode développement

Dans Android Studio :

```bash
# Build + installation sur un appareil/émulateur connecté
./gradlew installDebug

# Ou lancer directement depuis l'IDE (Run ▶) avec un appareil connecté en USB
```

L'application démarre sur l'écran caméra, capture une photo, exécute les 3 inférences TFLite en local, interroge l'API Groq en parallèle, fusionne les résultats et propose l'enregistrement du diagnostic sur Hedera.

## Contribution

### Flux de contribution

```bash
# 1. Créer une branche depuis main
git checkout -b feature/nom-de-la-fonctionnalite

# 2. Développer puis committer ses changements
git add .
git commit -m "feat: description du changement"

# 3. Pousser la branche
git push origin feature/nom-de-la-fonctionnalite

# 4. Ouvrir une Pull Request vers main et demander une revue
```

Merci de respecter les conventions de nommage de commits ([Conventional Commits](https://www.conventionalcommits.org/fr/)) et de documenter tout nouveau script ou paramètre ajouté dans les parties `ia/` ou `app_mobile/`.

## Construit avec

### Langages & Frameworks

- [Python](https://www.python.org/) — scripts de préparation et d'entraînement des modèles
- [TensorFlow / Keras](https://www.tensorflow.org/) — construction et entraînement des réseaux de neurones, transfer learning **MobileNetV2**
- [TensorFlow Lite](https://www.tensorflow.org/lite) — conversion et inférence embarquée des modèles sur mobile
- [scikit-learn](https://scikit-learn.org/) — normalisation, split des données, métriques d'évaluation
- [OpenCV](https://opencv.org/) — traitement d'images (redimensionnement, conversion couleur)
- [Java](https://developer.android.com/language) — développement de l'application Android
- [Android SDK / AndroidX AppCompat](https://developer.android.com/jetpack/androidx) — composants natifs de l'application
- [CameraX](https://developer.android.com/training/camerax) — capture photo et aperçu caméra
- [OkHttp](https://square.github.io/okhttp/) — appels réseau vers le backend (analyse LLM, blockchain)

### Outils

#### CI

_Aucune intégration continue n'est actuellement configurée dans le dépôt fourni._ À définir, par exemple :
- [GitHub Actions](https://docs.github.com/fr/actions) pour l'exécution automatisée des tests Python et du build Gradle Android
- Variables/secrets à prévoir : accès aux datasets ou artefacts de modèles, éventuelles clés d'API pour les tests

#### Déploiement

- Backend d'analyse LLM : API [Groq](https://console.groq.com/docs) — nécessite une clé d'API Groq côté serveur backend (non incluse dans ce dépôt mobile/IA)
- Traçabilité blockchain : réseau [Hedera](https://docs.hedera.com/hedera) — nécessite un compte Hedera (Testnet/Mainnet) et ses identifiants côté backend
- Distribution de l'application Android : à définir (ex. build APK/AAB via `./gradlew assembleRelease` puis publication manuelle ou via [Google Play Console](https://play.google.com/console/about/))

> Le dépôt fourni ne contient pas encore le code du backend exposant `/api/v1/analyze` et `/api/v1/blockchain/record` : ces informations sont à compléter une fois ce service documenté.

## Documentation

- [SDNET2018 – dataset de fissures](https://digitalcommons.usu.edu/all_datasets/48/)
- [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- [Documentation TensorFlow Lite](https://www.tensorflow.org/lite/guide)
- [Documentation CameraX](https://developer.android.com/training/camerax)
- [Documentation Groq API](https://console.groq.com/docs)
- [Documentation Hedera](https://docs.hedera.com/hedera)


## Licence

Voir le fichier [LICENSE](./LICENSE.md) du dépôt.

## Auteur

**Rimeh Abidi**
Contact : rimeh.abidi@enis.tn
