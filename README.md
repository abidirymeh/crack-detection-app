# PFA_ADEVA — Crack Detection & Building Diagnosis

## About

This project aims to make it easier to diagnose the condition of a building (facades, walls) directly from a smartphone. From a simple photo and a few details about the building's conditions, the application automatically detects the presence of **cracks**, assesses the **condition of the paint**, and estimates the **material strength** (concrete), using embedded Deep Learning models (TensorFlow Lite). The diagnosis can be refined through additional analysis via an **LLM (Groq)** and archived in a tamper-proof way on the **Hedera blockchain**, for the purpose of traceability in building maintenance.

## Demo

https://github.com/user-attachments/assets/0e760895-a8b4-46db-8153-72c51cff1c92

https://github.com/user-attachments/assets/faf0218d-afd5-4a6b-8c94-fd0988d1d70f

https://github.com/user-attachments/assets/3c3398fc-217d-461e-bd21-2cb2fe2957eb

https://github.com/user-attachments/assets/c7137b42-7a23-42f9-beb2-167002d84e0e

https://github.com/user-attachments/assets/1765306c-17e6-4762-b6f0-d0f9b21aa065

## Table of Contents

- 🪧 [About](#about)
- 📦 [Requirements](#requirements)
- 🚀 [Installation](#installation)
- 🛠️ [Usage](#usage)
- 🤝 [Contributing](#contributing)
- 🏗️ [Built With](#built-with)
- 📚 [Documentation](#documentation)
- 🏷️ [Versioning](#versioning)
- 📝 [License](#license)
- 👤 [Author](#author)

## Requirements

The project consists of two parts (`ia/` and `app_mobile/`), each with its own requirements.

**`ia/` part (model training):**

- [Python 3.9+](https://www.python.org/downloads/) — language used for all preparation and training scripts
- [TensorFlow](https://www.tensorflow.org/install) — building and training the neural networks (Keras)
- [scikit-learn](https://scikit-learn.org/stable/install.html) — train/test split, normalization (`StandardScaler`), metrics
- [OpenCV (opencv-python)](https://pypi.org/project/opencv-python/) — reading, resizing, and preprocessing images
- [NumPy](https://numpy.org/install/) / [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) — data manipulation
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) / [Seaborn](https://seaborn.pydata.org/installing.html) — visualization (distributions, correlations, training curves)
- [joblib](https://joblib.readthedocs.io/en/stable/) — saving/loading the scaler
- Datasets: [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) (crack images), [CCIC](https://data.mendeley.com/datasets/5y9wdsg2zt/2) (Concrete Crack Images for Classification), [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

**`app_mobile/` part (Android application):**

- [Android Studio](https://developer.android.com/studio) — development IDE
- [JDK 11+](https://developer.android.com/build/jdks) — building the Android project
- [Android SDK](https://developer.android.com/tools) (API level as specified in `build.gradle`) — Android build tools
- A physical Android device or emulator with a **camera**, since crack recognition requires a live shot
- Network access to a backend exposing:
  - an LLM analysis API (`/api/v1/analyze?model=groq`) — see [Groq documentation](https://console.groq.com/docs)
  - a blockchain recording API (`/api/v1/blockchain/record`) — see [Hedera documentation](https://docs.hedera.com/)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/abidirymeh/crack-detection-app.git
cd crack-detection-app
```

**2. Set up the Python environment (`ia/`)**

```bash
cd ia
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
```

**3. Download and place the datasets**

Download SDNET2018, CCIC, and UCI Concrete, then adjust the file paths (currently hardcoded, e.g. `C:/Users/.../03_dataset/`) at the top of the relevant scripts.

**4. Open the Android project (`app_mobile/`)**

```bash
cd ../app_mobile
```

Open the folder in Android Studio, let Gradle sync the dependencies (CameraX, TensorFlow Lite, OkHttp), then set the backend URL in `MainActivity.java` (the `API_BASE` constant).

**5. Copy the trained models into the app**

Once the `.tflite` models have been generated from the `ia/` side, place them in:

```bash
cp modele_enriched.tflite modele_concrete.tflite modele_peinture_agca.tflite \
   app_mobile/app/src/main/assets/
```

## Usage

### Train / regenerate the models (`ia/`)

```bash
# 1. Crack detection
python step1verifier_Structure_dataset.py
python step2preparerdonnee.py
python train.py                     # → modele_enriched.tflite

# 2. Material strength (real dataset recommended)
python 2b_preparer_uci_concrete.py
python 3b_entrainer_uci_concrete.py
python 6_convertir_tflite.py        # → modele_concrete.tflite
python scaler_params.py             # → scaler_params.json

# 3. Paint condition
python aug.py
python augmenter_dataset.py
python augmentert_peinture.py       # decision threshold calibration

# 4. Overall integration test (cracks + materials)
python 5_integration_fissures.py
```

### Run the app in development mode

In Android Studio:

```bash
# Build + install on a connected device/emulator
./gradlew installDebug

# Or run directly from the IDE (Run ▶) with a device connected via USB
```

The app starts on the camera screen, captures a photo, runs the 3 TFLite inferences locally, queries the Groq API in parallel, merges the results, and offers to record the diagnosis on Hedera.

## Contributing

### Contribution workflow

```bash
# 1. Create a branch from main
git checkout -b feature/feature-name

# 2. Make your changes, then commit them
git add .
git commit -m "feat: description of the change"

# 3. Push the branch
git push origin feature/feature-name

# 4. Open a Pull Request against main and request a review
```

Please follow the commit naming conventions ([Conventional Commits](https://www.conventionalcommits.org/)) and document any new script or parameter added in the `ia/` or `app_mobile/` parts.

## Built With

### Languages & Frameworks

- [Python](https://www.python.org/) — model preparation and training scripts
- [TensorFlow / Keras](https://www.tensorflow.org/) — building and training the neural networks, **MobileNetV2** transfer learning
- [TensorFlow Lite](https://www.tensorflow.org/lite) — conversion and embedded on-device inference of the models
- [scikit-learn](https://scikit-learn.org/) — normalization, data splitting, evaluation metrics
- [OpenCV](https://opencv.org/) — image processing (resizing, color conversion)
- [Java](https://developer.android.com/language) — Android app development
- [Android SDK / AndroidX AppCompat](https://developer.android.com/jetpack/androidx) — native app components
- [CameraX](https://developer.android.com/training/camerax) — photo capture and camera preview
- [OkHttp](https://square.github.io/okhttp/) — network calls to the backend (LLM analysis, blockchain)

### Tools

#### CI

_No continuous integration is currently configured in the provided repository._ To be defined, for example:
- [GitHub Actions](https://docs.github.com/en/actions) for automated Python tests and Android Gradle build execution
- Variables/secrets to plan for: access to datasets or model artifacts, any API keys needed for testing

#### Deployment

- LLM analysis backend: [Groq](https://console.groq.com/docs) API — requires a Groq API key on the backend server side (not included in this mobile/AI repository)
- Blockchain traceability: [Hedera](https://docs.hedera.com/hedera) network — requires a Hedera account (Testnet/Mainnet) and its credentials on the backend side
- Android app distribution: to be defined (e.g. APK/AAB build via `./gradlew assembleRelease` then manual publishing or via the [Google Play Console](https://play.google.com/console/about/))

> The provided repository does not yet contain the backend code exposing `/api/v1/analyze` and `/api/v1/blockchain/record`: these details are to be filled in once that service is documented.

## Documentation

- [SDNET2018 – crack dataset](https://digitalcommons.usu.edu/all_datasets/48/)
- [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- [TensorFlow Lite documentation](https://www.tensorflow.org/lite/guide)
- [CameraX documentation](https://developer.android.com/training/camerax)
- [Groq API documentation](https://console.groq.com/docs)
- [Hedera documentation](https://docs.hedera.com/hedera)

## License

See the repository's [LICENSE](./LICENSE.md) file.

## Author

**Rimeh Abidi**
Contact: rimeh.abidi@enis.tn
