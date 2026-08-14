package com.example.detectionfissuresapp2;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "CrackDetection";
    private static final String MODEL_FILE = "modele_enriched.tflite";
    private static final String MODEL_MATERIAUX = "modele_materiaux.tflite";
    private static final String MODEL_PEINTURE = "modele_peinture_agca.tflite";
    private static final int INPUT_SIZE = 224;

    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};
/*llm*/
    private static final String API_BASE       = "http://172.20.10.3:8000";
    private static final String API_ANALYZE    = API_BASE + "/api/v1/analyze?model=groq";
    private static final String API_BLOCKCHAIN = API_BASE + "/api/v1/blockchain/record";

    private volatile Boolean tfliteHasCrack = null;
    private volatile Boolean llmHasCrack    = null;
    private volatile float   tfliteProba    = 0f;

    private View cameraScreen;
    private View resultScreen;


    private PreviewView previewView;
    private ImageView capturedImage;
    private TextView resultText;
    private Button captureButton;
    private Button newPhotoButton;
    private Button analyserPeintureButton;
    private Button analyserMateriaux;
    private ScrollView resultPanel;
    private LinearLayout peinturePanel;
    private EditText ageBatiment, humidite, temperature, usure, nbFissures, distanceField;
    private TextView resultatMateriaux;
    private TextView resultatPeinture;

    // Models
    private Interpreter tflite;
    private Interpreter tfliteMateriaux;
    private Interpreter tflitePeinture;
    private ExecutorService cameraExecutor;
    private ImageCapture imageCapture;
    private Bitmap currentBitmap;

    private TextView resultLLM;
    private static final String API_URL = "http://172.20.10.3:8000/api/v1/analyze?model=groq";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        capturedImage = findViewById(R.id.capturedImage);
        resultText = findViewById(R.id.resultText);
        captureButton = findViewById(R.id.captureButton);
        newPhotoButton = findViewById(R.id.newPhotoButton);
        analyserPeintureButton = findViewById(R.id.analyserPeintureButton);
        analyserMateriaux = findViewById(R.id.analyserMateriaux);
        resultPanel = findViewById(R.id.resultPanel);
        peinturePanel = findViewById(R.id.peinturePanel);
        ageBatiment = findViewById(R.id.ageBatiment);
        humidite = findViewById(R.id.humidite);
        temperature = findViewById(R.id.temperature);
        usure = findViewById(R.id.usure);
        nbFissures = findViewById(R.id.nbFissures);
        resultatMateriaux = findViewById(R.id.resultatMateriaux);
        resultatPeinture = findViewById(R.id.resultatPeinture);
        distanceField = findViewById(R.id.distance);


        resultLLM = findViewById(R.id.resultLLM);
        cameraScreen = findViewById(R.id.cameraScreen);
        resultScreen = findViewById(R.id.resultScreen);
        cameraScreen.setVisibility(View.VISIBLE);
        resultScreen.setVisibility(View.GONE);

        captureButton.setOnClickListener(v -> captureImage());
        newPhotoButton.setOnClickListener(v -> resetToCamera());
        analyserMateriaux.setOnClickListener(v -> analyserMateriaux());
        analyserPeintureButton.setOnClickListener(v -> afficherFormulairePeinture());

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (allPermissionsGranted()) {
            initTensorFlowLite();
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void afficherFormulairePeinture() {
        if (currentBitmap == null) {
            Toast.makeText(this, "Prenez d'abord une photo", Toast.LENGTH_SHORT).show();
            return;
        }

        if (tflitePeinture == null) {
            Toast.makeText(this, "Modele peinture non disponible", Toast.LENGTH_SHORT).show();
            return;
        }

        if (peinturePanel != null) {
            resultPanel.setVisibility(View.VISIBLE);
            peinturePanel.setVisibility(View.VISIBLE);
        }

        String fissureResult = resultText.getText().toString();
        boolean fissureDetectee = fissureResult.contains("FISSURE") || fissureResult.contains("GRAVE") ||
                fissureResult.contains("MODEREE") || fissureResult.contains("LEGERE");

        analyserPeintureAutomatique(currentBitmap, fissureDetectee);
    }

    private void analyserPeintureAutomatique(Bitmap bitmap, boolean fissureDetectee) {
        try {
            ByteBuffer input = preprocessImage(bitmap);
            float[][] output = new float[1][1];

            tflitePeinture.run(input, output);
            float probability = output[0][0];
            float SEUIL_PEINTURE = 0.30f;

            boolean peintureAbimeeModele = probability < SEUIL_PEINTURE;

            String resultat;
            String recommandation;
            String budget;

            if (fissureDetectee) {
                resultat = "Peinture abimee (fissure detectee)";
                recommandation = "La presence de fissure indique que la peinture est endommagee.\n" +
                        "Un traitement de surface et une repainture sont necessaires.";
                budget = "Budget estime: 15-25€/m² (reparation + peinture)";
            } else if (peintureAbimeeModele) {
                resultat = "Peinture abimee";
                recommandation = "La peinture est endommagee. Un rafraichissement est necessaire.";
                budget = "Budget estime: 10-15€/m²";
            } else {
                resultat = "Peinture en bon etat";
                recommandation = "La peinture est en bon etat. Entretien normal suffisant.";
                budget = "Budget estime: 5-8€/m² (entretien)";
            }

            String finalResult = resultat + "\n\n" + recommandation + "\n" + budget;
            runOnUiThread(() -> resultatPeinture.setText(finalResult));

            Log.d(TAG, "Fissure: " + fissureDetectee + ", Peinture proba: " + probability);

        } catch (Exception e) {
            Log.e(TAG, "Erreur analyse peinture", e);
            runOnUiThread(() -> resultatPeinture.setText("Erreur lors de l'analyse de la peinture"));
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initTensorFlowLite();
                startCamera();
            } else {
                Toast.makeText(this, "Permission camera refusee", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void initTensorFlowLite() {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setUseXNNPACK(false);
            options.setNumThreads(1);

            tflite = new Interpreter(loadModelFile(MODEL_FILE), options);
            tfliteMateriaux = new Interpreter(loadModelFile(MODEL_MATERIAUX), options);

            try {
                tflitePeinture = new Interpreter(loadModelFile(MODEL_PEINTURE), options);
                Log.d(TAG, "Modele peinture charge");
            } catch (Exception e) {
                Log.w(TAG, "Modele peinture non trouve", e);
            }

            Log.d(TAG, "Modeles charges");
            runOnUiThread(() -> Toast.makeText(this, "Modeles charges", Toast.LENGTH_SHORT).show());

        } catch (Exception e) {
            Log.e(TAG, "Erreur chargement modeles", e);
            runOnUiThread(() -> Toast.makeText(this, "Erreur: " + e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    private MappedByteBuffer loadModelFile(String filename) throws IOException {
        try (AssetFileDescriptor fileDescriptor = getAssets().openFd(filename);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder().build();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                Log.d(TAG, "Camera demarree");

            } catch (Exception e) {
                Log.e(TAG, "Erreur demarrage camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {
        if (imageCapture == null) return;

        imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                currentBitmap = imageProxyToBitmap(image);

                runOnUiThread(() -> {
                    capturedImage.setImageBitmap(currentBitmap);
                    cameraScreen.setVisibility(View.GONE);
                    resultScreen.setVisibility(View.VISIBLE);
                    peinturePanel.setVisibility(View.GONE);
                });


                tfliteHasCrack = null;
                llmHasCrack    = null;
                tfliteProba    = 0f;
                detectCracksWithFusion(currentBitmap);
                analyserAvecLLMFusion(currentBitmap);


                image.close();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Erreur capture", exception);
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Erreur de capture", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    private void resetToCamera() {
        runOnUiThread(() -> {
            cameraScreen.setVisibility(View.VISIBLE);
            resultScreen.setVisibility(View.GONE);
            peinturePanel.setVisibility(View.GONE);
            resultatMateriaux.setVisibility(View.GONE);
            resultText.setText("");
            resultLLM.setText("Analyse LLM en attente...");
            resultatMateriaux.setText("");
            resultatPeinture.setText("");
            ageBatiment.setText("");
            humidite.setText("");
            temperature.setText("");
            usure.setText("");
            nbFissures.setText("");
            distanceField.setText("");
            currentBitmap = null;
            tfliteHasCrack = null;
            llmHasCrack    = null;
            tfliteProba    = 0f;
        });
    }

    private String construireMessage(float proba, float distance) {
        float SEUIL = 0.46f;
        StringBuilder result = new StringBuilder();

        // Message principal
        if (proba > SEUIL) {
            if (proba > 0.85f) {
                result.append(String.format(Locale.FRENCH,
                        "🔴🔴 FISSURE GRAVE !\nURGENT : Inspection immediate\n" +
                                "Confiance: %.1f%%\nDistance: %.1f m",
                        proba * 100, distance));
            } else if (proba > 0.70f) {
                result.append(String.format(Locale.FRENCH,
                        "🔴 FISSURE MODEREE\nSurveillance recommandee\n" +
                                "Confiance: %.1f%%\nDistance: %.1f m",
                        proba * 100, distance));
            } else {
                result.append(String.format(Locale.FRENCH,
                        "🟠 FISSURE LEGERE\nA surveiller\n" +
                                "Confiance: %.1f%%\nDistance: %.1f m",
                        proba * 100, distance));
            }
        } else {
            result.append(String.format(Locale.FRENCH,
                    "🟢 SURFACE SAINE\nAucune fissure detectee\n" +
                            "Confiance: %.1f%%\nDistance: %.1f m",
                    (1 - proba) * 100, distance));
        }

        if (distance > 5.0f) {
            result.append("\n\n⚠️ Distance > 5m : Detection moins fiable.\nRapprochez-vous pour confirmer.");
        }

        if (proba > 0.55f && proba < 0.65f && distance < 2.0f) {
            result.append("\n\n📷 Zone incertaine. Prenez une photo plus nette.");
        }

        return result.toString();
    }

    private void analyserMateriaux() {
        if (tfliteMateriaux == null) {
            runOnUiThread(() -> resultatMateriaux.setText("Modele materiaux non charge"));
            return;
        }

        try {
            float age = Float.parseFloat(ageBatiment.getText().toString());
            float hum = Float.parseFloat(humidite.getText().toString());
            float temp = Float.parseFloat(temperature.getText().toString());
            float wear = Float.parseFloat(usure.getText().toString());
            float cracks = Float.parseFloat(nbFissures.getText().toString());

            float[][] inputRaw = new float[1][5];
            inputRaw[0][0] = age;
            inputRaw[0][1] = hum;
            inputRaw[0][2] = temp;
            inputRaw[0][3] = wear;
            inputRaw[0][4] = cracks;

            float[] mean = {49.29f, 60.30f, 17.05f, 5.13f, 24.63f};
            float[] std = {29.48f, 16.83f, 13.19f, 2.82f, 14.58f};

            float[][] input = new float[1][5];
            for (int i = 0; i < 5; i++) {
                input[0][i] = (inputRaw[0][i] - mean[i]) / std[i];
            }


            float[][] output = new float[1][1];
            tfliteMateriaux.run(input, output);
            float resistance = output[0][0];


            String resultat;
            if (resistance < 20) {
                resultat = "RESISTANCE TRES FAIBLE : " + String.format(Locale.FRENCH, "%.1f", resistance) + " MPa\nRisque structurel eleve !";
            } else if (resistance < 30) {
                resultat = "RESISTANCE MODEREE : " + String.format(Locale.FRENCH, "%.1f", resistance) + " MPa\nSurveillance recommandee";
            } else if (resistance < 50) {
                resultat = "BONNE RESISTANCE : " + String.format(Locale.FRENCH, "%.1f", resistance) + " MPa\nStructure saine";
            } else {
                resultat = "RESISTANCE EXCELLENTE : " + String.format(Locale.FRENCH, "%.1f", resistance) + " MPa\nStructure de haute qualite";
            }

            runOnUiThread(() -> {
                resultatMateriaux.setText(resultat);
                resultatMateriaux.setVisibility(View.VISIBLE);
            });

        } catch (Exception e) {
            Log.e(TAG, "Erreur analyse materiaux", e);
            runOnUiThread(() -> resultatMateriaux.setText("Erreur: veuillez saisir des valeurs valides"));
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        ByteBuffer input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        input.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            input.putFloat(r);
            input.putFloat(g);
            input.putFloat(b);
        }

        return input;
    }

    private void analyserAvecLLMFusion(Bitmap bitmap) {
        runOnUiThread(() -> resultLLM.setText("⏳ Analyse LLM en cours..."));

        new Thread(() -> {
            try {
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, baos);
                byte[] imageBytes = baos.toByteArray();

                okhttp3.OkHttpClient client = new okhttp3.OkHttpClient.Builder()
                        .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
                        .readTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
                        .build();

                okhttp3.RequestBody requestBody = new okhttp3.MultipartBody.Builder()
                        .setType(okhttp3.MultipartBody.FORM)
                        .addFormDataPart("file", "image.jpg",
                                okhttp3.RequestBody.create(imageBytes,
                                        okhttp3.MediaType.parse("image/jpeg")))
                        .build();

                okhttp3.Request request = new okhttp3.Request.Builder()
                        .url(API_ANALYZE)
                        .post(requestBody)
                        .build();

                okhttp3.Response response = client.newCall(request).execute();

                if (response.isSuccessful()) {
                    String json   = response.body().string();
                    org.json.JSONObject obj = new org.json.JSONObject(json);
                    String report = obj.getString("report").replace("\\n", "\n");

                    llmHasCrack = parseLLMHasCrack(report);

                    final String finalReport = report;
                    runOnUiThread(() -> resultLLM.setText(finalReport));

                    tryFusionAndSendToBlockchain();

                } else {
                    final String err = "Erreur API LLM: " + response.code();
                    runOnUiThread(() -> resultLLM.setText(err));
                    llmHasCrack = tfliteHasCrack != null ? tfliteHasCrack : false;
                    tryFusionAndSendToBlockchain();
                }

            } catch (Exception e) {
                Log.e(TAG, "Erreur LLM", e);
                final String err = "Erreur connexion: " + e.getMessage()
                        + "\nVérifier que le serveur est lancé";
                runOnUiThread(() -> resultLLM.setText(err));
                llmHasCrack = tfliteHasCrack != null ? tfliteHasCrack : false;
                tryFusionAndSendToBlockchain();
            }
        }).start();
    }
    private boolean parseLLMHasCrack(String report) {
        if (report == null) return false;
        String lower = report.toLowerCase();

        int idx = lower.indexOf("détection");
        if (idx == -1) idx = lower.indexOf("detection");

        if (idx != -1) {
            String after = lower.substring(idx, Math.min(idx + 120, lower.length()));

            if (after.contains(": oui") || after.contains(":oui")  ||
                    after.contains("**oui") || after.contains("oui,")  ||
                    after.contains("oui —") || after.contains("oui -")) {
                return true;
            }
            if (after.contains(": non") || after.contains(":non") ||
                    after.contains("**non")) {
                return false;
            }
            if (after.contains("incertain")) {
                return true;
            }
        }

        return lower.contains("fissure")      || lower.contains("crack")      ||
                lower.contains("structurelle") || lower.contains("capillaire") ||
                lower.contains("critique")     || lower.contains("élevé")      ||
                lower.contains("eleve");
    }


    private synchronized void tryFusionAndSendToBlockchain() {
        if (tfliteHasCrack == null || llmHasCrack == null) return;


        boolean finalHasCrack;
        String  fusionSource;

        if (tfliteHasCrack.equals(llmHasCrack)) {
            finalHasCrack = tfliteHasCrack;
            fusionSource  = tfliteHasCrack ? "TFLite+LLM accordés: FISSURE"
                    : "TFLite+LLM accordés: SAIN";
        } else if (tfliteHasCrack) {
            finalHasCrack = true;
            fusionSource  = "Désaccord → TFLite détecte fissure (retenue)";
        } else {
            finalHasCrack = true;
            fusionSource  = "Désaccord → LLM détecte fissure (retenue)";
        }

        String detectedState = finalHasCrack ? "fissure_detectee" : "sain";
        String severity      = tfliteProba > 0.85f ? "critique"
                : tfliteProba > 0.70f ? "moderee"
                : tfliteProba > 0.46f ? "legere"
                : "aucune";

        String summary = (finalHasCrack
                ? "Fissure detectee. Severite: " + severity + ". "
                : "Surface saine. ")
                + "Source: " + fusionSource
                + ". TFLite: " + String.format(Locale.FRENCH, "%.1f%%", tfliteProba * 100);
        if (summary.length() > 200) summary = summary.substring(0, 197) + "...";
        final String reportSummary = summary;

        final String fusionMsg =
                "\n\n━━━━━━━━━━━━━━━━━━━━\n" +
                        "🔗 FUSION TFLite + LLM\n" +
                        fusionSource + "\n" +
                        "→ Résultat: " + detectedState.toUpperCase() + "\n" +
                        "⏳ Envoi Hedera...";

        runOnUiThread(() -> {
            String current = resultLLM.getText().toString();
            resultLLM.setText(current + fusionMsg);
        });

        tfliteHasCrack = null;
        llmHasCrack    = null;

        sendToHedera(detectedState, tfliteProba, severity, reportSummary);
    }
    private void sendToHedera(String detectedState, float confidence,
                              String severity, String reportSummary) {
        new Thread(() -> {
            try {
                String buildingId = "BAT-" + System.currentTimeMillis();

                org.json.JSONObject body = new org.json.JSONObject();
                body.put("building_id",    buildingId);
                body.put("filename",       "capture_" + buildingId + ".jpg");
                body.put("model_used",     "tflite+groq_fusion");
                body.put("detected_state", detectedState);
                body.put("confidence",     (double) confidence);
                body.put("severity",       severity);
                body.put("report_summary", reportSummary);

                okhttp3.OkHttpClient client = new okhttp3.OkHttpClient.Builder()
                        .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
                        .readTimeout(30,  java.util.concurrent.TimeUnit.SECONDS)
                        .build();

                okhttp3.Request request = new okhttp3.Request.Builder()
                        .url(API_BLOCKCHAIN)
                        .post(okhttp3.RequestBody.create(
                                body.toString(),
                                okhttp3.MediaType.parse("application/json; charset=utf-8")))
                        .build();

                okhttp3.Response response = client.newCall(request).execute();
                String responseJson = response.body().string();

                if (response.isSuccessful()) {
                    org.json.JSONObject result = new org.json.JSONObject(responseJson);

                    String txInfo = "";
                    if (result.has("transaction_id")) {
                        txInfo = "\n📋 Tx: " + result.getString("transaction_id");
                    } else if (result.has("topic_1")) {
                        try {
                            org.json.JSONObject t1 = result.getJSONObject("topic_1");
                            if (t1.has("transaction_id")) {
                                txInfo = "\n📋 Tx: " + t1.getString("transaction_id");
                            }
                        } catch (Exception ignored) {}
                    }

                    final String successMsg =
                            "\n✅ Enregistré sur Hedera !" + txInfo +
                                    "\n🏗️ ID: " + buildingId;

                    runOnUiThread(() -> {
                        String current = resultLLM.getText().toString();
                        resultLLM.setText(current + successMsg);
                    });

                } else {
                    final String errMsg = "\n❌ Hedera erreur " + response.code();
                    runOnUiThread(() -> {
                        String current = resultLLM.getText().toString();
                        resultLLM.setText(current + errMsg);
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Erreur Hedera", e);
                final String errMsg = "\n❌ Erreur Hedera: " + e.getMessage();
                runOnUiThread(() -> {
                    String current = resultLLM.getText().toString();
                    resultLLM.setText(current + errMsg);
                });
            }
        }).start();
    }

    private void detectCracksWithFusion(Bitmap bitmap) {
        if (bitmap == null) return;

        // Préparation des données pour la mesure
        ByteBuffer input = preprocessImage(bitmap);
        float[][] output = new float[1][1];
        float[][] inputMat = new float[1][5]; // Entrée bidon pour le test de performance

        // ============================================
        // MESURE INFÉRENCE + TEMPS DE RÉPONSE
        // ============================================

        long tempsReponseDebut = System.currentTimeMillis();

        // ===== INFÉRENCE FISSURES =====
        long inferenceDebut = System.nanoTime();
        if (tflite != null) tflite.run(input, output);
        long inferenceFin = System.nanoTime();
        double inferenceFissures = (inferenceFin - inferenceDebut) / 1_000_000.0;

        // ===== INFÉRENCE PEINTURE =====
        inferenceDebut = System.nanoTime();
        if (tflitePeinture != null) tflitePeinture.run(input, output);
        inferenceFin = System.nanoTime();
        double inferencePeinture = (inferenceFin - inferenceDebut) / 1_000_000.0;

        // ===== INFÉRENCE RÉSISTANCE =====
        inferenceDebut = System.nanoTime();
        if (tfliteMateriaux != null) tfliteMateriaux.run(inputMat, output);
        inferenceFin = System.nanoTime();
        double inferenceResistance = (inferenceFin - inferenceDebut) / 1_000_000.0;

        long tempsReponseFin = System.currentTimeMillis();
        double tempsReponseTotal = tempsReponseFin - tempsReponseDebut;

        // ===== AFFICHAGE DES PERFORMANCES =====
        String resultatPerf =
                "═══════════════════════════════\n" +
                        "📊 INFÉRENCE (modèle seul) :\n" +
                        "   Fissures   : " + String.format("%.0f", inferenceFissures) + " ms\n" +
                        "   Peinture   : " + String.format("%.0f", inferencePeinture) + " ms\n" +
                        "   Résistance : " + String.format("%.0f", inferenceResistance) + " ms\n" +
                        "═══════════════════════════════\n" +
                        "⏱️ TEMPS DE RÉPONSE (total) :\n" +
                        "   Application : " + String.format("%.0f", tempsReponseTotal) + " ms\n" +
                        "═══════════════════════════════";

        Toast.makeText(this, resultatPerf, Toast.LENGTH_LONG).show();
        Log.d("PERFORMANCES", resultatPerf);

        if (tflite == null) return;

        try {
            float distance = 1.0f;
            try {
                distance = Float.parseFloat(distanceField.getText().toString());
                if (distance <= 0) distance = 1.0f;
            } catch (Exception e) {
                distance = 1.0f;
            }

            // On utilise les données déjà préprocessées
            tflite.run(input, output);
            float probability = output[0][0];

            float probabilityCorrigee = probability;
            if (distance > 2.0f) {
                float correction = 1 + Math.min(0.15f, (distance - 2.0f) * 0.03f);
                probabilityCorrigee = probability * correction;
            } else if (distance < 0.8f) {
                float correction = 1 - Math.min(0.15f, (0.8f - distance) * 0.2f);
                probabilityCorrigee = probability * correction;
            }

            final float finalProba    = probabilityCorrigee;
            final float finalDistance = distance;

            tfliteProba    = finalProba;
            tfliteHasCrack = (finalProba > 0.46f);

            String message = construireMessage(finalProba, finalDistance);
            runOnUiThread(() -> resultText.setText(message));

            tryFusionAndSendToBlockchain();

        } catch (Exception e) {
            Log.e(TAG, "Erreur detection", e);
            runOnUiThread(() -> resultText.setText("Erreur de detection"));
        }
    }



    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) tflite.close();
        if (tfliteMateriaux != null) tfliteMateriaux.close();
        if (tflitePeinture != null) tflitePeinture.close();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}